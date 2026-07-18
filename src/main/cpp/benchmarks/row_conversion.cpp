/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/generate_input.hpp"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <nvbench/nvbench.cuh>
#include <row_conversion.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Benchmark grid for the four public row-conversion APIs. Eight benchmarks (to/from rows ×
// fixed/wide/strings/types) cover the Spark plugin's routing keys: the 100-column and 1536-byte
// fixed-width-optimized gates, multi-batch row output, the null-mask-absent branch, string
// schemas, and the full fixed-width type range. Every state estimates its device footprint up
// front and skips cells that do not fit, so one grid runs unchanged on any GPU size. Timed
// regions contain exactly the labeled conversion: to-rows cells run one conversion per
// iteration, from-rows cells convert pre-generated row batches back.

namespace {

enum class direction { to_rows, from_rows };

// The fixed-width-optimized kernels stage whole rows in the default 48 KB shared-memory budget
// with at least one 32-thread warp per block, so rows above 48 KB / 32 = 1536 bytes throw.
constexpr int64_t max_fixed_opt_row_bytes = 48 * 1024 / 32;

// Fraction of the free device memory one cell may claim; the remainder absorbs the conversion's
// smaller transient allocations and allocator slack.
constexpr double free_memory_fraction = 0.85;

// Conversions also allocate a few size_type work buffers that scale with the row count (per-row
// sizes and offsets, batch bounds); this per-row allowance bounds them.
constexpr int64_t per_row_metadata_bytes = 16;

// 9-type cycle of the pre-rewrite benchmark, kept because the grid's boundary arithmetic
// (~1400-byte rows at 256 columns, the 1536-byte crossing at 320) is derived on it.
std::vector<cudf::type_id> const default_cycle = {cudf::type_id::INT8,
                                                  cudf::type_id::INT32,
                                                  cudf::type_id::INT16,
                                                  cudf::type_id::INT64,
                                                  cudf::type_id::INT32,
                                                  cudf::type_id::BOOL8,
                                                  cudf::type_id::UINT16,
                                                  cudf::type_id::UINT8,
                                                  cudf::type_id::UINT64};

std::vector<cudf::type_id> const int_cycle = {cudf::type_id::INT8,
                                              cudf::type_id::INT16,
                                              cudf::type_id::INT32,
                                              cudf::type_id::INT64,
                                              cudf::type_id::UINT8,
                                              cudf::type_id::UINT16,
                                              cudf::type_id::UINT32,
                                              cudf::type_id::UINT64,
                                              cudf::type_id::BOOL8};

std::vector<cudf::type_id> const float_cycle = {cudf::type_id::FLOAT32, cudf::type_id::FLOAT64};

std::vector<cudf::type_id> const decimal_cycle = {
  cudf::type_id::DECIMAL32, cudf::type_id::DECIMAL64, cudf::type_id::DECIMAL128};

std::vector<cudf::type_id> const chrono_cycle = {cudf::type_id::TIMESTAMP_DAYS,
                                                 cudf::type_id::TIMESTAMP_SECONDS,
                                                 cudf::type_id::TIMESTAMP_MILLISECONDS,
                                                 cudf::type_id::TIMESTAMP_MICROSECONDS,
                                                 cudf::type_id::TIMESTAMP_NANOSECONDS,
                                                 cudf::type_id::DURATION_SECONDS,
                                                 cudf::type_id::DURATION_MILLISECONDS,
                                                 cudf::type_id::DURATION_MICROSECONDS,
                                                 cudf::type_id::DURATION_NANOSECONDS};

// Column count of the *_types benches: general path, just under the plugin's 100-column gate.
constexpr cudf::size_type types_bench_num_columns = 96;

// FNV-1a over the benchmark name and axis values. std::hash is not stable across standard
// libraries; an unstable seed would bench different data per build and inflate A/B gate noise.
constexpr uint64_t fnv1a(std::string_view text, uint64_t hash = 0xcbf29ce484222325ULL)
{
  for (unsigned char const c : text) {
    hash = (hash ^ c) * 0x100000001b3ULL;
  }
  return hash;
}

template <typename... AxisValues>
unsigned deterministic_seed(std::string_view bench_name, AxisValues const&... values)
{
  auto hash       = fnv1a(bench_name);
  auto const fold = [&hash](auto const& value) {
    hash = fnv1a("|", hash);
    if constexpr (std::is_arithmetic_v<std::remove_cvref_t<decltype(value)>>) {
      hash = fnv1a(std::to_string(value), hash);
    } else {
      hash = fnv1a(value, hash);
    }
  };
  (fold(values), ...);
  return static_cast<unsigned>(hash ^ (hash >> 32));
}

std::vector<cudf::data_type> make_schema(std::vector<cudf::type_id> const& types)
{
  std::vector<cudf::data_type> schema;
  schema.reserve(types.size());
  std::ranges::transform(
    types, std::back_inserter(schema), [](cudf::type_id id) { return cudf::data_type{id}; });
  return schema;
}

// Mirrors the JCUDF layout math in row_conversion.cu (compute_fixed_width_layout and
// compute_column_information): each fixed-width column packs at an offset aligned to its own
// element size, a string column packs an 8-byte offset/length pair at 4-byte alignment, and
// ceil(columns / 8) validity bytes follow unaligned.
int64_t jcudf_fixed_and_validity_bytes(std::vector<cudf::data_type> const& schema)
{
  int64_t offset = 0;
  for (auto const& type : schema) {
    bool const is_string = type.id() == cudf::type_id::STRING;
    auto const size      = is_string ? int64_t{8} : static_cast<int64_t>(cudf::size_of(type));
    auto const alignment = is_string ? int64_t{4} : size;
    offset               = (offset + alignment - 1) / alignment * alignment + size;
  }
  return offset + (static_cast<int64_t>(schema.size()) + 7) / 8;
}

// Complete fixed-width JCUDF row: data plus validity, padded to the 8-byte row alignment.
int64_t jcudf_row_size(std::vector<cudf::data_type> const& schema)
{
  return (jcudf_fixed_and_validity_bytes(schema) + 7) / 8 * 8;
}

// Bytes a conversion reads from a realized table: element data, string chars and offsets, and
// whichever null masks are actually present.
int64_t realized_table_bytes(cudf::table_view const& table)
{
  auto const mask_bytes = static_cast<int64_t>((table.num_rows() + 7) / 8);
  int64_t bytes         = 0;
  for (auto const& col : table) {
    if (col.type().id() == cudf::type_id::STRING) {
      auto const strings = cudf::strings_column_view{col};
      bytes += strings.chars_size(cudf::get_default_stream());
      bytes += static_cast<int64_t>(strings.offsets().size()) *
               static_cast<int64_t>(cudf::size_of(strings.offsets().type()));
    } else {
      bytes += static_cast<int64_t>(col.size()) * static_cast<int64_t>(cudf::size_of(col.type()));
    }
    if (col.nullable()) { bytes += mask_bytes; }
  }
  return bytes;
}

// Bytes convert_from_rows* writes when rebuilding this table: it always allocates a null mask
// per output column and rebuilds string offsets as int32, independent of the input's masks.
int64_t reconstructed_table_bytes(cudf::table_view const& table)
{
  auto const num_rows   = static_cast<int64_t>(table.num_rows());
  auto const mask_bytes = (num_rows + 7) / 8;
  int64_t bytes         = 0;
  for (auto const& col : table) {
    if (col.type().id() == cudf::type_id::STRING) {
      auto const strings = cudf::strings_column_view{col};
      bytes += strings.chars_size(cudf::get_default_stream());
      bytes += (num_rows + 1) * static_cast<int64_t>(sizeof(int32_t));
    } else {
      bytes += num_rows * static_cast<int64_t>(cudf::size_of(col.type()));
    }
    bytes += mask_bytes;
  }
  return bytes;
}

int64_t total_row_buffer_bytes(std::vector<std::unique_ptr<cudf::column>> const& row_batches)
{
  int64_t bytes = 0;
  for (auto const& batch : row_batches) {
    bytes += cudf::lists_column_view{batch->view()}.child().size();
  }
  return bytes;
}

// Estimated peak footprint of a fixed-width cell: input table (masks counted even when absent,
// because the from-rows reconstruction always allocates them) plus the JCUDF row buffer. The
// from-rows benches free the input table before timing, so the reconstruction output fits inside
// the same table-sized term; both directions therefore share this bound.
int64_t estimated_fixed_width_footprint(std::vector<cudf::data_type> const& schema,
                                        cudf::size_type num_rows)
{
  auto const rows      = static_cast<int64_t>(num_rows);
  int64_t data_per_row = 0;
  for (auto const& type : schema) {
    data_per_row += static_cast<int64_t>(cudf::size_of(type));
  }
  auto const table_bytes =
    rows * data_per_row + static_cast<int64_t>(schema.size()) * ((rows + 7) / 8);
  return table_bytes + rows * jcudf_row_size(schema) + rows * per_row_metadata_bytes;
}

// Estimated peak footprint of a strings cell. String lengths are NORMAL[0, 2·len] with mean len;
// nulls only shrink the realized sizes below this estimate. Row bytes add the per-row 8-byte
// alignment on top of the mean payload. from-rows additionally stages int32 offset and length
// work columns per string column while rebuilding.
int64_t estimated_strings_footprint(std::vector<cudf::data_type> const& schema,
                                    cudf::size_type num_rows,
                                    int64_t avg_string_len,
                                    cudf::size_type num_string_columns,
                                    direction dir)
{
  auto const rows            = static_cast<int64_t>(num_rows);
  auto const string_cols     = static_cast<int64_t>(num_string_columns);
  int64_t fixed_data_per_row = 0;
  for (auto const& type : schema) {
    if (type.id() != cudf::type_id::STRING) {
      fixed_data_per_row += static_cast<int64_t>(cudf::size_of(type));
    }
  }
  auto const chars_per_row = string_cols * avg_string_len;
  auto const table_bytes   = rows * (fixed_data_per_row + chars_per_row + 8 * string_cols) +
                           static_cast<int64_t>(schema.size()) * ((rows + 7) / 8);
  auto const row_buffer_bytes = rows * (jcudf_fixed_and_validity_bytes(schema) + chars_per_row + 8);
  auto const staging_bytes    = dir == direction::from_rows ? rows * 8 * string_cols : int64_t{0};
  return table_bytes + row_buffer_bytes + staging_bytes + rows * per_row_metadata_bytes;
}

// True (and marks the state skipped) when the estimate exceeds the free-memory budget queried at
// run time — never hardcoded, so larger-VRAM devices automatically activate more cells.
[[nodiscard]] bool exceeds_free_device_memory(nvbench::state& state, int64_t footprint_bytes)
{
  std::size_t free_bytes  = 0;
  std::size_t total_bytes = 0;
  CUDF_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (static_cast<double>(footprint_bytes) <=
      free_memory_fraction * static_cast<double>(free_bytes)) {
    return false;
  }
  state.skip("exceeds device memory");
  return true;
}

// Round-robin over the type categories so adjacent columns mix element sizes (1 to 16 bytes),
// exercising the row layout's per-column alignment padding.
std::vector<cudf::type_id> interleaved_mix(cudf::size_type num_columns)
{
  std::array const categories = {&int_cycle, &float_cycle, &decimal_cycle, &chrono_cycle};
  std::vector<cudf::type_id> out;
  out.reserve(num_columns);
  for (std::size_t i = 0; i < static_cast<std::size_t>(num_columns); ++i) {
    auto const& category = *categories[i % categories.size()];
    out.push_back(category[(i / categories.size()) % category.size()]);
  }
  return out;
}

std::vector<cudf::type_id> types_for_mix(std::string_view type_mix, cudf::size_type num_columns)
{
  if (type_mix == "ints") { return cycle_dtypes(int_cycle, num_columns); }
  if (type_mix == "floats") { return cycle_dtypes(float_cycle, num_columns); }
  if (type_mix == "decimals") { return cycle_dtypes(decimal_cycle, num_columns); }
  if (type_mix == "timestamps") { return cycle_dtypes(chrono_cycle, num_columns); }
  if (type_mix == "mixed") { return interleaved_mix(num_columns); }
  CUDF_FAIL("Unsupported type_mix axis value: " + std::string{type_mix});
}

void run_fixed_width_bench(nvbench::state& state,
                           direction dir,
                           bool use_fixed_opt,
                           cudf::size_type num_rows,
                           std::vector<cudf::type_id> const& types,
                           std::optional<double> null_probability,
                           unsigned seed)
{
  auto const schema   = make_schema(types);
  auto const row_size = jcudf_row_size(schema);
  if (use_fixed_opt && row_size > max_fixed_opt_row_bytes) {
    state.skip("row size exceeds the fixed-width-optimized 1536-byte limit");
    return;
  }
  if (exceeds_free_device_memory(state, estimated_fixed_width_footprint(schema, num_rows))) {
    return;
  }

  // Independent draws per element: the generator's default sample-pool-with-run-lengths mode
  // would repeat 2000 values in runs of ~4, an artifact this grid deliberately avoids.
  data_profile const profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(null_probability);
  auto table = create_random_table(types, row_count{num_rows}, profile, seed);

  // Decimal columns carry a generated scale, so the from-rows schema must come from the table.
  std::vector<cudf::data_type> realized_schema;
  realized_schema.reserve(types.size());
  for (auto const& col : table->view()) {
    realized_schema.push_back(col.type());
  }

  // Exact for fixed-width schemas: every row occupies the same padded size in every batch.
  auto const row_buffer_bytes = row_size * num_rows;
  state.add_element_count(num_rows, "rows");

  if (dir == direction::to_rows) {
    state.add_global_memory_reads<int8_t>(realized_table_bytes(table->view()));
    state.add_global_memory_writes<int8_t>(row_buffer_bytes);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      auto const row_batches =
        use_fixed_opt
          ? spark_rapids_jni::convert_to_rows_fixed_width_optimized(table->view(), stream)
          : spark_rapids_jni::convert_to_rows(table->view(), stream);
    });
  } else {
    auto const row_batches =
      use_fixed_opt ? spark_rapids_jni::convert_to_rows_fixed_width_optimized(table->view())
                    : spark_rapids_jni::convert_to_rows(table->view());
    state.add_global_memory_reads<int8_t>(row_buffer_bytes);
    state.add_global_memory_writes<int8_t>(reconstructed_table_bytes(table->view()));
    table.reset();  // the timed loop needs only the row batches and the schema
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      for (auto const& batch : row_batches) {
        cudf::lists_column_view const list{batch->view()};
        auto const out = use_fixed_opt
                           ? spark_rapids_jni::convert_from_rows_fixed_width_optimized(
                               list, realized_schema, stream)
                           : spark_rapids_jni::convert_from_rows(list, realized_schema, stream);
      }
    });
  }
}

void run_strings_bench(nvbench::state& state, direction dir, std::string_view bench_name)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const avg_string_len = state.get_int64("avg_string_len");
  auto const string_cols    = static_cast<cudf::size_type>(state.get_int64("string_cols"));
  auto const seed           = deterministic_seed(bench_name, num_rows, avg_string_len, string_cols);

  // Eight fixed-width columns (leading types of the default cycle) followed by string columns.
  std::vector<cudf::type_id> types(default_cycle.begin(), default_cycle.begin() + 8);
  types.insert(types.end(), string_cols, cudf::type_id::STRING);
  auto const schema = make_schema(types);

  auto const footprint =
    estimated_strings_footprint(schema, num_rows, avg_string_len, string_cols, dir);
  if (exceeds_free_device_memory(state, footprint)) { return; }

  // Independent draws per element (see run_fixed_width_bench); for strings a repeated sample
  // pool would additionally make the char-copy kernels artificially cache-hot.
  data_profile const profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(0.1).distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, int64_t{0}, 2 * avg_string_len);
  auto table = create_random_table(types, row_count{num_rows}, profile, seed);

  std::vector<cudf::data_type> realized_schema;
  realized_schema.reserve(types.size());
  for (auto const& col : table->view()) {
    realized_schema.push_back(col.type());
  }

  // One up-front conversion supplies the exact row-buffer size (per-row string payloads make it
  // data-dependent) and, for from-rows, the timed input batches.
  auto row_batches            = spark_rapids_jni::convert_to_rows(table->view());
  auto const row_buffer_bytes = total_row_buffer_bytes(row_batches);
  state.add_element_count(num_rows, "rows");

  if (dir == direction::to_rows) {
    state.add_global_memory_reads<int8_t>(realized_table_bytes(table->view()));
    state.add_global_memory_writes<int8_t>(row_buffer_bytes);
    row_batches.clear();  // keep each iteration's transient row buffer as the only live copy
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      auto const out    = spark_rapids_jni::convert_to_rows(table->view(), stream);
    });
  } else {
    state.add_global_memory_reads<int8_t>(row_buffer_bytes);
    state.add_global_memory_writes<int8_t>(reconstructed_table_bytes(table->view()));
    table.reset();  // the timed loop needs only the row batches and the schema
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      for (auto const& batch : row_batches) {
        cudf::lists_column_view const list{batch->view()};
        auto const out = spark_rapids_jni::convert_from_rows(list, realized_schema, stream);
      }
    });
  }
}

}  // namespace

static void to_rows_fixed(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const path        = state.get_string("path");
  run_fixed_width_bench(state,
                        direction::to_rows,
                        path == "fixed_opt",
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        0.1,
                        deterministic_seed("to_rows_fixed", num_rows, num_columns, path));
}

static void from_rows_fixed(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const path        = state.get_string("path");
  run_fixed_width_bench(state,
                        direction::from_rows,
                        path == "fixed_opt",
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        0.1,
                        deterministic_seed("from_rows_fixed", num_rows, num_columns, path));
}

static void to_rows_wide(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const nulls       = state.get_string("nulls");
  auto const null_probability =
    nulls == "none" ? std::optional<double>{} : std::optional<double>{std::stod(nulls)};
  run_fixed_width_bench(state,
                        direction::to_rows,
                        false,
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        null_probability,
                        deterministic_seed("to_rows_wide", num_rows, num_columns, nulls));
}

static void from_rows_wide(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const nulls       = state.get_string("nulls");
  auto const null_probability =
    nulls == "none" ? std::optional<double>{} : std::optional<double>{std::stod(nulls)};
  run_fixed_width_bench(state,
                        direction::from_rows,
                        false,
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        null_probability,
                        deterministic_seed("from_rows_wide", num_rows, num_columns, nulls));
}

static void to_rows_strings(nvbench::state& state)
{
  run_strings_bench(state, direction::to_rows, "to_rows_strings");
}

static void from_rows_strings(nvbench::state& state)
{
  run_strings_bench(state, direction::from_rows, "from_rows_strings");
}

static void to_rows_types(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type_mix = state.get_string("type_mix");
  run_fixed_width_bench(state,
                        direction::to_rows,
                        false,
                        num_rows,
                        types_for_mix(type_mix, types_bench_num_columns),
                        0.1,
                        deterministic_seed("to_rows_types", num_rows, type_mix));
}

static void from_rows_types(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type_mix = state.get_string("type_mix");
  run_fixed_width_bench(state,
                        direction::from_rows,
                        false,
                        num_rows,
                        types_for_mix(type_mix, types_bench_num_columns),
                        0.1,
                        deterministic_seed("from_rows_types", num_rows, type_mix));
}

NVBENCH_BENCH(to_rows_fixed)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {2, 10, 96, 128, 212, 256})
  .add_string_axis("path", {"general", "fixed_opt"});

NVBENCH_BENCH(from_rows_fixed)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {2, 10, 96, 128, 212, 256})
  .add_string_axis("path", {"general", "fixed_opt"});

NVBENCH_BENCH(to_rows_wide)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {212, 320})
  .add_string_axis("nulls", {"none", "0.1"});

NVBENCH_BENCH(from_rows_wide)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {212, 320})
  .add_string_axis("nulls", {"none", "0.1"});

NVBENCH_BENCH(to_rows_strings)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("avg_string_len", {16, 128})
  .add_int64_axis("string_cols", {2, 8});

NVBENCH_BENCH(from_rows_strings)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("avg_string_len", {16, 128})
  .add_int64_axis("string_cols", {2, 8});

NVBENCH_BENCH(to_rows_types)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_string_axis("type_mix", {"ints", "floats", "decimals", "timestamps", "mixed"});

NVBENCH_BENCH(from_rows_types)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_string_axis("type_mix", {"ints", "floats", "decimals", "timestamps", "mixed"});
