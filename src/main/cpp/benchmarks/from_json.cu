/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <json_utils.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Default string value width (mirrors get_json_object.cu's min/max width of 10) so the generated
// JSON stays simple. Individual benchmarks may override the width via the `value_width` argument of
// `generate_input`.
constexpr auto default_value_width = 10;

// Build a list of `num_keys` all-STRING column types. The raw-map engine extracts every value as a
// raw substring, so a homogeneous all-STRING flat object is the representative input.
std::vector<cudf::type_id> make_all_string_column_types(int num_keys)
{
  return std::vector<cudf::type_id>(num_keys, cudf::type_id::STRING);
}

// Generate a strings column where each row is one FLAT JSON object whose scalar fields take the
// caller-supplied `column_types`. The number of keys is `column_types.size()`.
//
// `create_random_table` is deterministic (fixed default seed = 1), so the same arguments always
// yield the same input.
//
// Tunable parameters:
//   - `value_width`  : exact width of every generated STRING value (NORMAL dist, lower==upper).
//   - `null_pct`     : if > 0, fraction of null elements per column; if 0, no null mask at all.
//   - `key_name_len` : if <= 0, keys are "col0".."colN-1"; if > 0, keys are EXACTLY `key_name_len`
//                      chars ("k" + zero-padded i) so longer key names can be exercised.
//   - `row_count`    : if > 0, the table is sized by row count and `size_bytes` is ignored; else
//   the table is sized by target byte count via `table_size_bytes`.
//
// `write_json(lines=true)` appends a trailing newline, so splitting on "\n" produces a trailing
// empty string (a spurious extra row). We slice it off below before returning.
std::unique_ptr<cudf::column> generate_input(std::size_t size_bytes,
                                             std::vector<cudf::type_id> const& column_types,
                                             int value_width       = default_value_width,
                                             double null_pct       = 0.0,
                                             int key_name_len      = 0,
                                             std::size_t row_count = 0)
{
  auto const num_keys = static_cast<int>(column_types.size());

  auto profile_builder = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, value_width, value_width);
  if (null_pct > 0.0) {
    profile_builder.null_probability(null_pct);
  } else {
    profile_builder.no_validity();
  }
  data_profile const table_profile = std::move(profile_builder);

  // `row_count` (the parameter) shadows the global `row_count` size-tag struct, so the
  // table-by-rows overload must name the tag explicitly via `::row_count`.
  auto const input_table =
    row_count > 0
      ? create_random_table(
          column_types, ::row_count{static_cast<cudf::size_type>(row_count)}, table_profile)
      : create_random_table(column_types, table_size_bytes{size_bytes}, table_profile);

  // JSON object keys must match the schema column names. With the default naming the keys are
  // "col0".."colN-1"; when `key_name_len > 0` the keys become "k" + zero-padded index, padded to
  // exactly `key_name_len` characters.
  std::vector<cudf::io::column_name_info> column_names(num_keys);
  if (key_name_len <= 0) {
    for (int i = 0; i < num_keys; ++i) {
      column_names[i].name = "col" + std::to_string(i);
    }
  } else {
    auto const num_digits = key_name_len - 1;
    CUDF_EXPECTS(num_digits > 0, "key_name_len must be at least 2 to hold 'k' plus one digit");
    // The largest index (num_keys - 1) must fit in `num_digits` decimal digits without truncation.
    auto max_representable = std::size_t{1};
    for (int d = 0; d < num_digits && max_representable < static_cast<std::size_t>(num_keys); ++d) {
      max_representable *= 10;
    }
    CUDF_EXPECTS(static_cast<std::size_t>(num_keys) <= max_representable,
                 "key_name_len has too few digits to keep all key names unique");
    std::vector<char> name_buf(static_cast<std::size_t>(key_name_len) + 1);
    for (int i = 0; i < num_keys; ++i) {
      std::snprintf(name_buf.data(), name_buf.size(), "k%0*d", num_digits, i);
      column_names[i].name = std::string{name_buf.data()};
    }
  }

  std::vector<char> buffer;
  cudf::io::sink_info sink(&buffer);
  cudf::io::table_metadata mt;
  mt.schema_info = std::move(column_names);
  auto write_opts =
    cudf::io::json_writer_options::builder(sink, input_table->view()).lines(true).metadata(mt);
  cudf::io::write_json(write_opts);

  // Split one JSON buffer into separate per-row JSON objects.
  auto const json_str = std::string{buffer.begin(), buffer.end()};
  auto const json_col = cudf::test::strings_column_wrapper{{json_str}};
  auto split_strs =
    cudf::strings::split_record(cudf::strings_column_view{json_col}, cudf::string_scalar("\n"))
      ->release();

  // split_strs is a LIST<STRING>; extract the child strings column.
  auto json_strings = std::move(split_strs.children[cudf::lists_column_view::child_column_index]);

  // The trailing newline from write_json yields a final empty string. Drop it by slicing the
  // strings column to [0, size - 1).
  auto const sliced = cudf::slice(json_strings->view(), {0, json_strings->size() - 1}).front();
  return std::make_unique<cudf::column>(sliced);
}

// Number of input bytes actually fed to an engine: the size of the strings column's character
// buffer. Used for `add_global_memory_reads` accounting on the row-shape sweep, where the input
// size is row-driven rather than declared via a `size_bytes` axis.
std::size_t input_char_bytes(cudf::column_view const& json_strings)
{
  auto const stream = cudf::get_default_stream();
  return static_cast<std::size_t>(cudf::strings_column_view{json_strings}.chars_size(stream));
}

// Upper bound for the deterministically varied width of every generated key and array-element
// string. A constexpr so the range is trivially adjustable.
constexpr cudf::size_type max_string_width = 20;

// SplitMix64 finalizer: a cheap, well-distributed hash used to spread token widths across
// [1, max_string_width] so generated keys and elements are not all the same length.
__device__ inline std::uint64_t hash_index(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Builds one JSON object row `{"<key>":["<elem>",...],...}` for the map-of-array engine. Every key
// and element is a quoted token whose width varies in [1, max_string_width] and whose content is a
// deterministic function of its index. A value is the literal `null` (null inner list, mask #1) at
// `value_null_stride`; an element is the literal `null` (null element, mask #2) at
// `element_null_stride`. The same code path computes the row size (when `d_chars == nullptr`) and
// writes the bytes, so the two passes cannot disagree.
struct map_of_array_row_fn {
  cudf::size_type keys_per_row;
  cudf::size_type array_len;
  cudf::size_type value_null_stride;    // 0 disables; else value null when index % stride == 0
  cudf::size_type element_null_stride;  // 0 disables; else element null when index % stride == 0

  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  // Elements draw from a different content stream than keys so a key never mirrors its own element.
  static constexpr std::uint64_t element_seed_salt = 0x1000003ULL;

  __device__ void operator()(cudf::size_type row) const
  {
    char* out         = d_chars ? d_chars + d_offsets[row] : nullptr;
    cudf::size_type n = 0;

    auto emit = [&](char c) {
      if (out) { out[n] = c; }
      ++n;
    };
    auto emit_literal = [&](char const* s) {
      for (; *s != '\0'; ++s) {
        emit(*s);
      }
    };
    // A quoted token of 1..max_string_width lowercase chars, seeded deterministically by `seed`.
    auto emit_token = [&](std::uint64_t seed) {
      auto const width = static_cast<cudf::size_type>(1 + hash_index(seed) % max_string_width);
      emit('"');
      for (cudf::size_type c = 0; c < width; ++c) {
        emit(static_cast<char>('a' + (seed + c) % 26));
      }
      emit('"');
    };

    emit('{');
    for (cudf::size_type k = 0; k < keys_per_row; ++k) {
      if (k > 0) { emit(','); }
      auto const key_index = static_cast<std::uint64_t>(row) * keys_per_row + k;
      emit_token(key_index);
      emit(':');
      if (value_null_stride != 0 && key_index % value_null_stride == 0) {
        emit_literal("null");
        continue;
      }
      emit('[');
      for (cudf::size_type e = 0; e < array_len; ++e) {
        if (e > 0) { emit(','); }
        auto const element_index = key_index * array_len + e;
        if (element_null_stride != 0 && element_index % element_null_stride == 0) {
          emit_literal("null");
        } else {
          emit_token(element_index + element_seed_salt);
        }
      }
      emit(']');
    }
    emit('}');

    if (out == nullptr) { d_sizes[row] = n; }
  }
};

// Generate a strings column where each row is one JSON object mapping `keys_per_row` keys to JSON
// arrays of `array_len` strings, targeting `from_json_to_raw_map_array_values`. Key and element
// widths vary in [1, max_string_width]. `element_null_pct` / `value_null_pct` inject literal `null`
// elements (mask #2) / values, i.e. null inner lists (mask #1), at a fixed cadence. The column is
// built entirely on the GPU, so large row counts stay cheap.
std::unique_ptr<cudf::column> generate_map_of_array_input(std::size_t num_rows,
                                                          int keys_per_row,
                                                          int array_len,
                                                          double element_null_pct = 0.0,
                                                          double value_null_pct   = 0.0)
{
  // Convert a fraction in [0,1] to a deterministic "1 in stride" cadence; 0 disables nulls.
  auto const to_stride = [](double pct) {
    return pct > 0.0 ? std::max(1, static_cast<int>(1.0 / pct)) : 0;
  };

  map_of_array_row_fn fn{static_cast<cudf::size_type>(keys_per_row),
                         static_cast<cudf::size_type>(array_len),
                         static_cast<cudf::size_type>(to_stride(value_null_pct)),
                         static_cast<cudf::size_type>(to_stride(element_null_pct))};

  auto const stream     = cudf::get_default_stream();
  auto const mr         = cudf::get_current_device_resource_ref();
  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    fn, static_cast<cudf::size_type>(num_rows), stream, mr);
  return cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                   std::move(offsets),
                                   chars.release(),
                                   0,
                                   rmm::device_buffer{});
}

}  // namespace

// Map engine: token-walk parser producing the LIST<STRUCT<STRING,STRING>> raw-map output.
void BM_from_json_to_raw_map(nvbench::state& state)
{
  auto const size_bytes = static_cast<std::size_t>(state.get_int64("size_bytes"));
  auto const num_keys   = static_cast<int>(state.get_int64("num_keys"));

  auto const json_strings = generate_input(size_bytes, make_all_string_column_types(num_keys));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

// Scenario: raw_map vs string value width.
void BM_from_json_to_raw_map_value_width(nvbench::state& state)
{
  constexpr std::size_t size_bytes = 10'000'000;
  constexpr int num_keys           = 8;
  auto const value_width           = static_cast<int>(state.get_int64("value_width"));

  auto const json_strings =
    generate_input(size_bytes, make_all_string_column_types(num_keys), value_width);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

// Scenario: raw_map vs null density.
void BM_from_json_to_raw_map_null_density(nvbench::state& state)
{
  constexpr std::size_t size_bytes = 10'000'000;
  constexpr int num_keys           = 8;
  auto const null_pct              = state.get_float64("null_pct");

  auto const json_strings = generate_input(
    size_bytes, make_all_string_column_types(num_keys), default_value_width, null_pct);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

// Scenario: raw_map at sub-megabyte input sizes (fixed-overhead / launch-bound behavior).
void BM_from_json_to_raw_map_micro_size(nvbench::state& state)
{
  auto const size_bytes  = static_cast<std::size_t>(state.get_int64("size_bytes"));
  constexpr int num_keys = 8;

  auto const json_strings = generate_input(size_bytes, make_all_string_column_types(num_keys));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

// Scenario: raw_map across row shapes at roughly constant total bytes.
void BM_from_json_to_raw_map_row_shape(nvbench::state& state)
{
  constexpr int num_keys = 8;
  auto const shape       = state.get_string("shape");

  std::size_t row_count_ = 0;
  int value_width        = default_value_width;
  if (shape == "few_wide") {
    row_count_  = 2'000;
    value_width = 2'000;
  } else if (shape == "balanced") {
    row_count_  = 40'000;
    value_width = 100;
  } else {  // "many_narrow"
    row_count_  = 4'000'000;
    value_width = 1;
  }

  auto const json_strings = generate_input(/*size_bytes=*/0,
                                           make_all_string_column_types(num_keys),
                                           value_width,
                                           /*null_pct=*/0.0,
                                           /*key_name_len=*/0,
                                           row_count_);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(input_char_bytes(json_strings->view()));
}

// Scenario: raw_map vs key-name length.
void BM_from_json_to_raw_map_key_name_len(nvbench::state& state)
{
  constexpr std::size_t size_bytes = 10'000'000;
  constexpr int num_keys           = 8;
  auto const key_name_len          = static_cast<int>(state.get_int64("key_name_len"));

  auto const json_strings = generate_input(size_bytes,
                                           make_all_string_column_types(num_keys),
                                           default_value_width,
                                           /*null_pct=*/0.0,
                                           key_name_len);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

// Map-of-array engine: parses MapType[String, Array[String]] JSON into the
// LIST<STRUCT<STRING, LIST<STRING>>> raw-map-of-array output.

// HEADLINE: map-of-array vs array length and row count.
void BM_from_json_to_raw_map_array_values(nvbench::state& state)
{
  auto const array_len       = static_cast<int>(state.get_int64("array_len"));
  auto const num_rows        = static_cast<std::size_t>(state.get_int64("num_rows"));
  constexpr int keys_per_row = 5;

  auto const json_strings = generate_map_of_array_input(num_rows, keys_per_row, array_len);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map_array_values(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(input_char_bytes(json_strings->view()));
}

// Scenario: map-of-array vs null density. One float axis drives BOTH the element-null mask and the
// value-null (null inner list) mask.
void BM_from_json_to_raw_map_array_values_null_density(nvbench::state& state)
{
  auto const null_pct            = state.get_float64("null_pct");
  constexpr std::size_t num_rows = 1'000'000;
  constexpr int array_len        = 3;
  constexpr int keys_per_row     = 5;

  auto const json_strings =
    generate_map_of_array_input(num_rows, keys_per_row, array_len, null_pct, null_pct);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map_array_values(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(input_char_bytes(json_strings->view()));
}

// Scenario: map-of-array vs keys (map entries) per row.
void BM_from_json_to_raw_map_array_values_keys_per_row(nvbench::state& state)
{
  auto const keys_per_row        = static_cast<int>(state.get_int64("keys_per_row"));
  constexpr std::size_t num_rows = 1'000'000;
  constexpr int array_len        = 3;

  auto const json_strings = generate_map_of_array_input(num_rows, keys_per_row, array_len);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::from_json_to_raw_map_array_values(
      cudf::strings_column_view{json_strings->view()}, false, true, true, false);
  });
  state.add_global_memory_reads<nvbench::int8_t>(input_char_bytes(json_strings->view()));
}

NVBENCH_BENCH(BM_from_json_to_raw_map)
  .set_name("from_json_to_raw_map")
  .add_int64_axis("size_bytes", {1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_keys", {1, 8, 32, 64});

NVBENCH_BENCH(BM_from_json_to_raw_map_value_width)
  .set_name("from_json_to_raw_map_value_width")
  .add_int64_axis("value_width", {1, 10, 100, 1'000});

NVBENCH_BENCH(BM_from_json_to_raw_map_null_density)
  .set_name("from_json_to_raw_map_null_density")
  .add_float64_axis("null_pct", {0.0, 0.1, 0.5, 0.9});

NVBENCH_BENCH(BM_from_json_to_raw_map_micro_size)
  .set_name("from_json_to_raw_map_micro_size")
  .add_int64_axis("size_bytes", {1'000, 10'000, 100'000, 1'000'000});

NVBENCH_BENCH(BM_from_json_to_raw_map_row_shape)
  .set_name("from_json_to_raw_map_row_shape")
  .add_string_axis("shape", {"few_wide", "balanced", "many_narrow"});

NVBENCH_BENCH(BM_from_json_to_raw_map_key_name_len)
  .set_name("from_json_to_raw_map_key_name_len")
  .add_int64_axis("key_name_len", {3, 16, 64});

NVBENCH_BENCH(BM_from_json_to_raw_map_array_values)
  .set_name("from_json_to_raw_map_array_values")
  .add_int64_axis("array_len", {1, 3, 10})
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000});

NVBENCH_BENCH(BM_from_json_to_raw_map_array_values_null_density)
  .set_name("from_json_to_raw_map_array_values_null_density")
  .add_float64_axis("null_pct", {0.0, 0.1, 0.5, 0.9});

NVBENCH_BENCH(BM_from_json_to_raw_map_array_values_keys_per_row)
  .set_name("from_json_to_raw_map_array_values_keys_per_row")
  .add_int64_axis("keys_per_row", {2, 5, 10});
