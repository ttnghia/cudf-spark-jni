#!/bin/bash
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Hermetic tests for the LIBCUDF_REUSE_PREBUILT path: the fingerprint skew guard, the phase-1/2
# build skips, the phase-3 stale-cache wipe, the reuse early-exits in submodule-check and
# [un]apply-patches, and build-info's install-tree resolution. No GPU, network, or real toolchain:
# cmake/nvcc/gcc/git/cuobjdump are PATH stubs that log their arguments, and the real build scripts
# run against fake prebuilt trees in a temporary directory. Requires gawk (as build-info does).
#
#   ./build/test-reuse-prebuilt.sh
#

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/reuse-prebuilt-test.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

#
# Tool stubs. Each logs its invocation to $STUB_LOG so tests can assert which commands ran.
#
STUB_BIN="$TEST_ROOT/bin"
mkdir -p "$STUB_BIN"

# A configure rewrites CMakeCache.txt with the -D cache entries (as STRING, matching the real
# cudf-spark-jni cache) so the trio staleness check greps realistic lines; --build only logs.
cat > "$STUB_BIN/cmake" <<'EOF'
#!/bin/bash
echo "cmake $*" >> "${STUB_LOG:-/dev/null}"
if [[ "${1:-}" == "--build" ]]; then exit 0; fi
: > CMakeCache.txt
for a in "$@"; do
  case "$a" in
    -DCUDF_DIR=*) echo "CUDF_DIR:STRING=${a#-DCUDF_DIR=}" >> CMakeCache.txt ;;
    -DCUDF_INSTALL_DIR=*) echo "CUDF_INSTALL_DIR:STRING=${a#-DCUDF_INSTALL_DIR=}" >> CMakeCache.txt ;;
    -DCUDFJNI_BUILD_DIR=*) echo "CUDFJNI_BUILD_DIR:STRING=${a#-DCUDFJNI_BUILD_DIR=}" >> CMakeCache.txt ;;
  esac
done
EOF

# rev-parse HEAD serves the sha from <dir>/.fake-sha (absent = not a repo); submodule status and
# rev-parse --show-toplevel are steered by STUB_SUBMODULE_OUTDATED / STUB_GIT_TOPLEVEL.
cat > "$STUB_BIN/git" <<'EOF'
#!/bin/bash
echo "git $*" >> "${STUB_LOG:-/dev/null}"
dir=.
if [[ "${1:-}" == "-C" ]]; then dir="$2"; shift 2; fi
case "$*" in
  "rev-parse HEAD")
    if [[ -f "$dir/.fake-sha" ]]; then cat "$dir/.fake-sha"; else
      echo "fatal: not a git repository" >&2; exit 128; fi ;;
  "rev-parse --abbrev-ref HEAD") echo "fake-branch" ;;
  "rev-parse --show-toplevel")
    if [[ -n "${STUB_GIT_TOPLEVEL:-}" ]]; then echo "$STUB_GIT_TOPLEVEL"; else
      echo "fatal: not a git repository" >&2; exit 128; fi ;;
  "config --get remote.origin.url") echo "https://github.com/example/fake.git" ;;
  "submodule status")
    if [[ -n "${STUB_SUBMODULE_OUTDATED:-}" ]]; then echo "-00deadbeef thirdparty/cudf"; fi ;;
  *) : ;;
esac
EOF

cat > "$STUB_BIN/nvcc" <<'EOF'
#!/bin/bash
echo "nvcc $*" >> "${STUB_LOG:-/dev/null}"
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 12.8, V12.8.61"
EOF

cat > "$STUB_BIN/gcc" <<'EOF'
#!/bin/bash
echo "gcc $*" >> "${STUB_LOG:-/dev/null}"
echo "13.2.0"
EOF

cat > "$STUB_BIN/cuobjdump" <<'EOF'
#!/bin/bash
echo "cuobjdump $*" >> "${STUB_LOG:-/dev/null}"
printf 'Fatbin elf code:\n================\narch = sm_80\ncode version = [1,7]\narch = sm_90\n'
EOF

chmod +x "$STUB_BIN"/*
export PATH="$STUB_BIN:$PATH"

#
# Test framework: one fixture directory per case; failures are collected, never fatal.
#
CASES=0
FAILS=0
CASE=""
CASE_OK=1

new_case() {
  CASE="$1"
  CASES=$((CASES + 1))
  CASE_OK=1
  CASE_DIR="$TEST_ROOT/case-$CASES"
  BASE="$CASE_DIR/base"           # fake PROJECT_BASE_DIR (this repo)
  BUILD="$CASE_DIR/target"        # fake PROJECT_BUILD_DIR
  CUDF_SRC="$CASE_DIR/cudf"       # fake cudf checkout
  PINS="$CASE_DIR/pins"           # fake thirdparty/cudf-pins
  INSTALL="$CASE_DIR/install"     # prebuilt libcudf install tree
  JNIBUILD="$CASE_DIR/libcudfjni" # prebuilt libcudfjni build tree
  SPARKJNI="$BUILD/jni/cmake-build"
  M1="$INSTALL/cudf-prebuilt-fingerprint.txt"
  M2="$JNIBUILD/cudf-prebuilt-fingerprint.txt"
  STUB_LOG="$CASE_DIR/stub.log"
  OUT="$CASE_DIR/out.log"
  export STUB_LOG
  mkdir -p "$BASE/src/main/cpp" "$BUILD" "$CUDF_SRC/java/src/main/native" "$CUDF_SRC/cpp" "$PINS"
  printf 'pin-a\n' > "$PINS/versions.json"
  printf 'pin-b\n' > "$PINS/setup.cmake"
  printf 'pin-c\n' > "$PINS/add_dependency_pins.cmake"
  printf 'pin-d\n' > "$PINS/rapids-cmake.sha"
  echo "aaaa1111" > "$CUDF_SRC/.fake-sha"
  : > "$STUB_LOG"
}

note_fail() {
  echo "  FAIL($CASE): $*"
  FAILS=$((FAILS + 1))
  CASE_OK=0
}

end_case() { [[ $CASE_OK -eq 1 ]] && echo "PASS: $CASE"; return 0; }

assert_rc0() { [[ "$RC" -eq 0 ]] || note_fail "expected exit 0, got $RC ($(tail -2 "$OUT" | tr '\n' ' '))"; }
assert_rc_nonzero() { [[ "$RC" -ne 0 ]] || note_fail "expected non-zero exit"; }
assert_in() { grep -qF -- "$1" "$2" || note_fail "missing '$1' in ${2##*/}"; }
assert_not_in() { grep -qF -- "$1" "$2" && note_fail "unexpected '$1' in ${2##*/}"; return 0; }
assert_file() { [[ -e "$1" ]] || note_fail "missing file: $1"; }
assert_no_file() { [[ -e "$1" ]] && note_fail "file should not exist: $1"; return 0; }

# Run the real buildcpp.sh against the case fixture; extra VAR=VAL args override the defaults
# (later env assignments win). Ambient reuse/toolkit variables are dropped for hermeticity.
run_buildcpp() {
  RC=0
  env -u FROM_MAVEN -u REUSE_ENV -u GPU_ARCHS -u CUDA_VERSION \
      -u STUB_SUBMODULE_OUTDATED -u STUB_GIT_TOPLEVEL \
      PROJECT_BASE_DIR="$BASE" PROJECT_BUILD_DIR="$BUILD" \
      CUDF_PATH="$CUDF_SRC" CUDF_PIN_PATH="$PINS" \
      LIBCUDF_INSTALL_PATH="$INSTALL" LIBCUDFJNI_BUILD_PATH="$JNIBUILD" \
      BUILD_TESTS=OFF BUILD_BENCHMARKS=OFF USE_GDS=OFF \
      CMAKE_CUDA_ARCHITECTURES=90 STUB_LOG="$STUB_LOG" \
      "$@" bash "$SCRIPT_DIR/buildcpp.sh" > "$OUT" 2>&1 || RC=$?
}

# Seed the prebuilt pair the way a real run does: a normal (non-reuse) stubbed build stamps both
# fingerprint manifests via buildcpp.sh's own stamp function, then the library artifacts the
# stubs never produce are faked in.
seed_prebuilt() {
  run_buildcpp LIBCUDF_REUSE_PREBUILT=false
  [[ "$RC" -eq 0 ]] || { note_fail "seed build failed rc=$RC ($(tail -2 "$OUT" | tr '\n' ' '))"; return 1; }
  mv "$OUT" "$CASE_DIR/seed.log"
  cp "$STUB_LOG" "$CASE_DIR/seed-stub.log"
  mkdir -p "$INSTALL/lib64" "$JNIBUILD/_deps/arrow-build/release"
  touch "$INSTALL/lib64/libcudf.a" "$JNIBUILD/libcudfjni.a" \
    "$JNIBUILD/_deps/arrow-build/release/libarrow.a" "$JNIBUILD/libnvcomp.so"
}

run_reuse() { : > "$STUB_LOG"; run_buildcpp LIBCUDF_REUSE_PREBUILT=true "$@"; }

# Run one of the real guard scripts (submodule-check / apply-patches / unapply-patches) from a
# directory the stub git does not treat as a repo; extra VAR=VAL args set the guard's environment.
run_script() {
  local script="$1"
  shift
  RC=0
  ( cd "$CASE_DIR" && env -u STUB_SUBMODULE_OUTDATED -u STUB_GIT_TOPLEVEL -u LIBCUDF_REUSE_PREBUILT \
      STUB_LOG="$STUB_LOG" "$@" bash "$SCRIPT_DIR/$script" ) > "$OUT" 2>&1 || RC=$?
}

run_build_info() {
  RC=0
  : > "$STUB_LOG"
  env STUB_LOG="$STUB_LOG" bash "$SCRIPT_DIR/build-info" "$@" > "$OUT" 2>&1 || RC=$?
}

#
# Fingerprint gate: seed and reuse from identical env agree, and both manifests carry the eight
# enforced cmp. dimensions plus one shared seed nonce.
#
new_case "reuse-pass-identical-env"
if seed_prebuilt; then
  assert_file "$M1"
  assert_file "$M2"
  for key in pins_sha256 ptds cuda_archs use_gds rmm_logging build_type dep_mode cuda_toolkit; do
    assert_in "cmp.$key=" "$M1"
    assert_in "cmp.$key=" "$M2"
  done
  [[ "$(grep -c '^cmp\.' "$M1")" -eq 8 ]] || note_fail "expected exactly 8 cmp. lines in $M1"
  assert_in "cmp.cuda_archs=90" "$M1"
  [[ "$(grep '^seed_nonce=' "$M1")" == "$(grep '^seed_nonce=' "$M2")" ]] \
    || note_fail "seed_nonce differs between manifests of one seed run"
  run_reuse
  assert_rc0
  assert_in "Skipping libcudf build" "$OUT"
  assert_in "Skipping libcudfjni build" "$OUT"
fi
end_case

# The reuse artifact check accepts the lib/ (non-lib64) install layout.
new_case "reuse-pass-lib-variant"
if seed_prebuilt; then
  mkdir -p "$INSTALL/lib"
  mv "$INSTALL/lib64/libcudf.a" "$INSTALL/lib/libcudf.a"
  rmdir "$INSTALL/lib64"
  run_reuse
  assert_rc0
fi
end_case

# One skewed cmp. dimension (CUDA archs here) must be rejected...
new_case "skew-cuda-archs-rejected"
if seed_prebuilt; then
  run_reuse CMAKE_CUDA_ARCHITECTURES=100
  assert_rc_nonzero
  assert_in "prebuilt does not match this worktree (skew in" "$OUT"
fi
end_case

# ...unless LIBCUDF_REUSE_FORCE downgrades the mismatch to a warning.
new_case "skew-forced-warns-and-builds"
if seed_prebuilt; then
  run_reuse CMAKE_CUDA_ARCHITECTURES=100 LIBCUDF_REUSE_FORCE=true
  assert_rc0
  assert_in "WARNING: cudf skew in" "$OUT"
  assert_in "overridden" "$OUT"
fi
end_case

# A changed cudf-pins file flips cmp.pins_sha256 and must be rejected.
new_case "skew-pins-rejected"
if seed_prebuilt; then
  echo "drifted" >> "$PINS/versions.json"
  run_reuse
  assert_rc_nonzero
  assert_in "prebuilt does not match this worktree (skew in" "$OUT"
fi
end_case

# A different CUDA toolkit (CUDA_VERSION vs the seed's nvcc-derived release) must be rejected.
new_case "skew-cuda-toolkit-rejected"
if seed_prebuilt; then
  run_reuse CUDA_VERSION=13.0
  assert_rc_nonzero
  assert_in "prebuilt does not match this worktree (skew in" "$OUT"
fi
end_case

# Manifests from different seed runs (mismatched pair) are refused even with identical cmp. lines.
new_case "nonce-mismatch-rejected"
if seed_prebuilt; then
  sed -i 's/^seed_nonce=.*/seed_nonce=00000000-dead-beef-0000-000000000000/' "$M2"
  run_reuse
  assert_rc_nonzero
  assert_in "prebuilt from different seed runs" "$OUT"
fi
end_case

new_case "nonce-mismatch-forced-warns"
if seed_prebuilt; then
  sed -i 's/^seed_nonce=.*/seed_nonce=00000000-dead-beef-0000-000000000000/' "$M2"
  run_reuse LIBCUDF_REUSE_FORCE=true
  assert_rc0
  assert_in "WARNING: install/libcudfjni from different seed runs" "$OUT"
fi
end_case

# Each required prebuilt artifact is checked with its own error message.
new_case "missing-libcudf-a-rejected"
if seed_prebuilt; then
  rm "$INSTALL/lib64/libcudf.a"
  run_reuse
  assert_rc_nonzero
  assert_in "ERROR: no prebuilt lib{64,}/libcudf.a under" "$OUT"
fi
end_case

new_case "missing-libcudfjni-a-rejected"
if seed_prebuilt; then
  rm "$JNIBUILD/libcudfjni.a"
  run_reuse
  assert_rc_nonzero
  assert_in "ERROR: no prebuilt libcudfjni.a at" "$OUT"
fi
end_case

new_case "missing-arrow-deps-rejected"
if seed_prebuilt; then
  rm "$JNIBUILD/_deps/arrow-build/release/libarrow.a"
  run_reuse
  assert_rc_nonzero
  assert_in "ERROR: incomplete libcudfjni _deps (arrow)" "$OUT"
fi
end_case

new_case "missing-nvcomp-rejected"
if seed_prebuilt; then
  rm "$JNIBUILD/libnvcomp.so"
  run_reuse
  assert_rc_nonzero
  assert_in "ERROR: prebuilt libcudfjni missing libnvcomp.so" "$OUT"
fi
end_case

# Tests/benchmarks additionally require the installed test utilities in the prebuilt.
new_case "missing-testutil-rejected"
if seed_prebuilt; then
  run_reuse BUILD_TESTS=ON
  assert_rc_nonzero
  assert_in "needs installed testutil" "$OUT"
fi
end_case

new_case "missing-testutil-stream-lib-rejected"
if seed_prebuilt; then
  mkdir -p "$INSTALL/src/cudftestutil"
  run_reuse BUILD_BENCHMARKS=ON
  assert_rc_nonzero
  assert_in "needs cudftest_default_stream" "$OUT"
fi
end_case

# Reuse refuses non-pinned dependency modes outright, force or not.
new_case "dep-mode-latest-rejected"
if seed_prebuilt; then
  run_reuse LIBCUDF_DEPENDENCY_MODE=latest
  assert_rc_nonzero
  assert_in "ERROR: reuse requires pinned deps; got latest" "$OUT"
fi
end_case

# An unstamped prebuilt (no manifest) and a manifest without a seed nonce are both refused.
new_case "manifest-missing-rejected"
if seed_prebuilt; then
  rm "$M1"
  run_reuse
  assert_rc_nonzero
  assert_in "ERROR: prebuilt not stamped" "$OUT"
fi
end_case

new_case "manifest-without-nonce-rejected"
if seed_prebuilt; then
  grep -v '^seed_nonce=' "$M2" > "$M2.tmp" && mv "$M2.tmp" "$M2"
  run_reuse
  assert_rc_nonzero
  assert_in "missing seed_nonce (corrupt/old manifest)" "$OUT"
fi
end_case

# Unhashable pins (broken CUDF_PIN_PATH) fail the fingerprint recompute with a clear error.
new_case "unreadable-pins-rejected"
if seed_prebuilt; then
  mkdir -p "$CASE_DIR/empty-pins"
  run_reuse CUDF_PIN_PATH="$CASE_DIR/empty-pins"
  assert_rc_nonzero
  assert_in "ERROR: cannot hash the tracked cudf-pins files" "$OUT"
fi
end_case

# cudf source skew (cudf.path HEAD != the prebuilt's recorded commit) warns but never blocks.
new_case "cudf-sha-skew-warns-only"
if seed_prebuilt; then
  echo "bbbb2222" > "$CUDF_SRC/.fake-sha"
  run_reuse
  assert_rc0
  assert_in "WARNING: cudf.path HEAD" "$OUT"
fi
end_case

# Reuse must skip the phase-1 libcudf and phase-2 libcudfjni cmake work entirely, while the seed
# run drives all three phases; only the phase-3 cudf-spark-jni configure+build may run.
new_case "reuse-skips-phase1-and-phase2"
if seed_prebuilt; then
  assert_in "cmake $CUDF_SRC/cpp" "$CASE_DIR/seed-stub.log"
  assert_in "--target install" "$CASE_DIR/seed-stub.log"
  assert_in "cmake $CUDF_SRC/java/src/main/native" "$CASE_DIR/seed-stub.log"
  assert_in "cmake --build $JNIBUILD" "$CASE_DIR/seed-stub.log"
  assert_in "cmake --build $SPARKJNI" "$CASE_DIR/seed-stub.log"
  run_reuse
  assert_rc0
  assert_not_in "$CUDF_SRC/cpp" "$STUB_LOG"
  assert_not_in "--target install" "$STUB_LOG"
  assert_not_in "$CUDF_SRC/java/src/main/native" "$STUB_LOG"
  assert_not_in "cmake --build $JNIBUILD" "$STUB_LOG"
  assert_in "cmake $BASE/src/main/cpp" "$STUB_LOG"
  assert_in "cmake --build $SPARKJNI" "$STUB_LOG"
fi
end_case

# Phase-3 staleness wipe: an unchanged CUDF_PATH/LIBCUDF_INSTALL_PATH/LIBCUDFJNI_BUILD_PATH trio
# must NOT wipe the build dir (idempotent rebuild), and a changed trio member must.
new_case "trio-unchanged-does-not-wipe"
if seed_prebuilt; then
  run_reuse
  assert_rc0
  assert_file "$SPARKJNI/CMakeCache.txt"
  touch "$SPARKJNI/survives-rebuild.marker"
  run_reuse
  assert_rc0
  assert_not_in "wiping" "$OUT"
  assert_file "$SPARKJNI/survives-rebuild.marker"
fi
end_case

new_case "trio-changed-wipes"
if seed_prebuilt; then
  run_reuse
  assert_rc0
  touch "$SPARKJNI/survives-rebuild.marker"
  mkdir -p "$CASE_DIR/cudf2"
  run_reuse CUDF_PATH="$CASE_DIR/cudf2"
  assert_rc0
  assert_in "wiping" "$OUT"
  assert_no_file "$SPARKJNI/survives-rebuild.marker"
  assert_in "CUDF_DIR:STRING=$CASE_DIR/cudf2" "$SPARKJNI/CMakeCache.txt"
fi
end_case

# Reuse early-exits: each guarded script must exit 0 under LIBCUDF_REUSE_PREBUILT=true in a
# directory where its real work would fail, and keep failing there when the flag is off.
new_case "submodule-check-reuse-skips"
run_script submodule-check LIBCUDF_REUSE_PREBUILT=true STUB_SUBMODULE_OUTDATED=1
assert_rc0
assert_not_in "git" "$STUB_LOG"
end_case

new_case "submodule-check-fails-without-flag"
run_script submodule-check STUB_SUBMODULE_OUTDATED=1
assert_rc_nonzero
assert_in "ERROR: submodules out of date" "$OUT"
run_script submodule-check LIBCUDF_REUSE_PREBUILT=false STUB_SUBMODULE_OUTDATED=1
assert_rc_nonzero
end_case

new_case "apply-patches-reuse-skips"
run_script apply-patches LIBCUDF_REUSE_PREBUILT=true
assert_rc0
assert_not_in "git" "$STUB_LOG"
end_case

new_case "apply-patches-fails-without-flag"
run_script apply-patches
assert_rc_nonzero
end_case

new_case "unapply-patches-reuse-skips"
run_script unapply-patches LIBCUDF_REUSE_PREBUILT=true
assert_rc0
assert_not_in "git" "$STUB_LOG"
end_case

new_case "unapply-patches-fails-without-flag"
run_script unapply-patches
assert_rc_nonzero
end_case

# build-info resolves an install tree to lib64/libcudf.a, falls back to lib/, errors clearly when
# neither exists, and passes a direct library file through unchanged.
new_case "build-info-lib64"
mkdir -p "$CASE_DIR/repo" "$CASE_DIR/tree/lib64"
echo "cafe0001" > "$CASE_DIR/repo/.fake-sha"
touch "$CASE_DIR/tree/lib64/libcudf.a"
run_build_info 25.02.0 "$CASE_DIR/repo" "$CASE_DIR/tree" cudf_sha=zzz
assert_rc0
assert_in "version=25.02.0" "$OUT"
assert_in "revision=cafe0001" "$OUT"
assert_in "gpu_architectures=80;90" "$OUT"
assert_in "cudf_sha=zzz" "$OUT"
assert_in "cuobjdump $CASE_DIR/tree/lib64/libcudf.a" "$STUB_LOG"
end_case

new_case "build-info-lib-fallback"
mkdir -p "$CASE_DIR/repo" "$CASE_DIR/tree/lib"
echo "cafe0002" > "$CASE_DIR/repo/.fake-sha"
touch "$CASE_DIR/tree/lib/libcudf.a"
run_build_info 25.02.0 "$CASE_DIR/repo" "$CASE_DIR/tree"
assert_rc0
assert_in "cuobjdump $CASE_DIR/tree/lib/libcudf.a" "$STUB_LOG"
end_case

new_case "build-info-neither-errors"
mkdir -p "$CASE_DIR/repo" "$CASE_DIR/tree"
echo "cafe0003" > "$CASE_DIR/repo/.fake-sha"
run_build_info 25.02.0 "$CASE_DIR/repo" "$CASE_DIR/tree"
assert_rc_nonzero
assert_in "ERROR: no prebuilt lib{64,}/libcudf.a under $CASE_DIR/tree" "$OUT"
assert_not_in "cuobjdump" "$STUB_LOG"
end_case

new_case "build-info-file-passthrough"
mkdir -p "$CASE_DIR/repo"
echo "cafe0004" > "$CASE_DIR/repo/.fake-sha"
touch "$CASE_DIR/libcudf.a"
run_build_info 25.02.0 "$CASE_DIR/repo" "$CASE_DIR/libcudf.a"
assert_rc0
assert_in "cuobjdump $CASE_DIR/libcudf.a" "$STUB_LOG"
end_case

echo
echo "$CASES cases, $FAILS failure(s)"
[[ "$FAILS" -eq 0 ]]
