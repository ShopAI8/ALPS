#!/bin/bash

# ==============================================================================
# build_hybrid.sh - Build UNG, ACORN, FAVOR, and NaviX indexes in one place.
#
# Responsibilities:
# 1. Compile the UNG, ACORN, FAVOR, and NaviX codebases.
# 2. Check whether `fvecs` data needs to be converted to `bin`.
# 3. Build indexes according to `build_mode` (`parallel`, `serial`, `ung_only`, `acorn_only`, `favor_only`, `navix_only`, `skip`, `compile`).
#    `skip` bypasses all compilation, data conversion, and index building for query-only runs.
# ==============================================================================

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

# Use a recent cmake (NaviX requires >=3.24, system has 3.22)
# Prefer the pip-installed one in ~/.local/bin over the system /usr/bin.
if [ -x "$HOME/.local/bin/cmake" ]; then
    CMAKE="$HOME/.local/bin/cmake"
elif [ -x "$HOME/.cache/uv/builds-v0/.tmprqNNhP/bin/cmake" ]; then
    CMAKE="$HOME/.cache/uv/builds-v0/.tmprqNNhP/bin/cmake"
else
    CMAKE=cmake
fi

# --- Step 1: parse command-line arguments ---
PARAMS=("$@")
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')

        if [[ $key == "BUILD_MODE" ]]; then
            BUILD_MODE="$2"
            shift 2
            continue
        fi

        if [ -z "$2" ]; then
            echo "错误: 参数 $1 缺少值"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "未知参数: $1"
        exit 1
    fi
done

case "$BUILD_MODE" in
    parallel|serial|ung_only|acorn_only|favor_only|navix_only|all|skip|compile)
        echo "[INFO] Build mode set to: $BUILD_MODE"
        ;;
    *)
        echo "错误: 无效的 build_mode '$BUILD_MODE'。可用选项: parallel, serial, ung_only, acorn_only, favor_only, navix_only, all, skip, compile"
        exit 1
        ;;
esac

# `skip` mode: bypass every build step (compilation, data conversion, index building).
# Indexes are assumed to already exist from a previous build run; the caller should
# proceed directly to ground-truth generation and search.
if [[ "$BUILD_MODE" == "skip" ]]; then
    echo "[INFO] Build mode is 'skip'. Skipping compilation, data conversion, and index building."
    echo "[SUCCESS] No build performed. Assuming indexes already exist; proceed directly to query."
    exit 0
fi

# --- Step 2: compile code ---

# 1. Build NaviX first because UNG depends on it.
if [[ -z "${NAVIX_BUILD_DIR:-}" ]]; then
    NAVIX_BUILD_DIR="${SCRIPT_DIR}/NaviX/build"
fi

if [ ! -f "${NAVIX_BUILD_DIR}/faiss_navix/libfaiss.a" ] && [ ! -f "${NAVIX_BUILD_DIR}/faiss_navix/libfaiss_avx2.a" ]; then
    echo "[INFO] NaviX library not found. Compiling..."
    rm -rf "$NAVIX_BUILD_DIR"
    mkdir -p "$NAVIX_BUILD_DIR"
    $CMAKE -S "${SCRIPT_DIR}/NaviX" -B "$NAVIX_BUILD_DIR" \
        -DFAISS_OPT_LEVEL=avx2 \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_EXTRAS=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release
    make -C "$NAVIX_BUILD_DIR" -j"$BUILD_JOBS"
else
    echo "[INFO] NaviX library found."
fi

# 2. Build UNG next. Knowhere is required.
if [[ -z "${KNOWHERE_INCLUDE_DIR:-}" || -z "${KNOWHERE_LIBRARY:-}" ]]; then
    echo "[WARN] Knowhere paths not set. Milvus baseline will be disabled."
fi

KNOWHERE_BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/build_knowhere.sh"
LOCAL_KNOWHERE_LIBRARY="${SCRIPT_DIR}/knowhere/build/Release/libknowhere.so"
if [[ ! -f "${KNOWHERE_LIBRARY}" && "${KNOWHERE_LIBRARY}" == "${LOCAL_KNOWHERE_LIBRARY}" && -f "${KNOWHERE_BOOTSTRAP_SCRIPT}" ]]; then
    echo "[INFO] Local knowhere library not found. Bootstrapping build via ${KNOWHERE_BOOTSTRAP_SCRIPT} ..."
    bash "${KNOWHERE_BOOTSTRAP_SCRIPT}"
fi

KNOWHERE_AVAILABLE=true
if [[ ! -f "${KNOWHERE_INCLUDE_DIR}/knowhere/index/index_factory.h" ]]; then
    echo "[WARN] Knowhere include not found: ${KNOWHERE_INCLUDE_DIR}"
    KNOWHERE_AVAILABLE=false
fi
if [[ ! -f "${KNOWHERE_LIBRARY}" ]]; then
    echo "[WARN] Knowhere library not found: ${KNOWHERE_LIBRARY}"
    KNOWHERE_AVAILABLE=false
fi

if $KNOWHERE_AVAILABLE; then
    echo "[INFO] Knowhere found. Milvus baseline enabled."
    KNOWHERE_FLAG="-DENABLE_KNOWHERE_MILVUS_BASELINE=ON"
else
    echo "[INFO] Knowhere not available. Milvus baseline disabled."
    KNOWHERE_FLAG="-DENABLE_KNOWHERE_MILVUS_BASELINE=OFF"
fi

if [[ -z "${UNG_BUILD_DIR:-}" ]]; then
    UNG_BUILD_DIR="${SCRIPT_DIR}/UNG/codes/build"
fi

mkdir -p "$UNG_BUILD_DIR"
$CMAKE -S "${SCRIPT_DIR}/UNG/codes" -B "$UNG_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DNAVIX_BUILD_DIR="$NAVIX_BUILD_DIR" \
    ${KNOWHERE_FLAG} \
    -DKNOWHERE_INCLUDE_DIR="${KNOWHERE_INCLUDE_DIR}" \
    -DKNOWHERE_LIBRARY="${KNOWHERE_LIBRARY}"
make -C "$UNG_BUILD_DIR" -j"$BUILD_JOBS"

UNG_EXECUTABLE="${UNG_BUILD_DIR}/apps/build_UNG_index"
if [[ ! -x "$UNG_EXECUTABLE" ]]; then
    echo "[ERROR] UNG executable not found after compilation: $UNG_EXECUTABLE"
    exit 1
fi

# 3. Build ACORN.
if [[ -z "${ACORN_BUILD_DIR:-}" ]]; then
    ACORN_BUILD_DIR="${SCRIPT_DIR}/ACORN/build"
fi

ACORN_EXECUTABLE="${ACORN_BUILD_DIR}/demos/test_acorn"
if [ ! -f "$ACORN_EXECUTABLE" ]; then
    echo "[INFO] ACORN executable not found. Compiling..."
    mkdir -p "$ACORN_BUILD_DIR"
    $CMAKE -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/tmp/googletest_src
    make -C "$ACORN_BUILD_DIR" -j"$BUILD_JOBS" test_acorn
else
    echo "[INFO] ACORN executable found."
fi

# 4. Build FAVOR.
if [[ -z "${FAVOR_BUILD_DIR:-}" ]]; then
    FAVOR_BUILD_DIR="${SCRIPT_DIR}/FAVOR/build"
fi

FAVOR_EXECUTABLE="${FAVOR_BUILD_DIR}/app/build_index"
FAVOR_BUILD_SOURCE="${SCRIPT_DIR}/FAVOR/app/build_index.cpp"
FAVOR_SOURCE_DIR="${SCRIPT_DIR}/FAVOR"

if [ ! -f "$FAVOR_EXECUTABLE" ] || [ "$FAVOR_BUILD_SOURCE" -nt "$FAVOR_EXECUTABLE" ]; then
    if [ ! -f "$FAVOR_EXECUTABLE" ]; then
        echo "[INFO] FAVOR executable not found. Compiling..."
    else
        echo "[INFO] FAVOR source is newer than executable. Recompiling..."
    fi
    echo "[INFO] Recreating FAVOR build directory before configuration..."
    rm -rf "$FAVOR_BUILD_DIR"
    mkdir -p "$FAVOR_BUILD_DIR"
    $CMAKE -S "$FAVOR_SOURCE_DIR" -B "$FAVOR_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -C "$FAVOR_BUILD_DIR" -j"$BUILD_JOBS" build_index
else
    echo "[INFO] FAVOR executable found."
fi

# --- Step 2.5: validate and convert the data format ---
echo "[INFO] Checking and converting data format (if necessary)..."
FVECS_TO_BIN_TOOL="${UNG_BUILD_DIR}/tools/fvecs_to_bin"
BASE_FVECS_FILE="${DATA_DIR}/${DATASET}_base.fvecs"
BASE_BIN_FILE="${DATA_DIR}/${DATASET}_base.bin"

if [ ! -x "$FVECS_TO_BIN_TOOL" ]; then
    echo "Error: Conversion tool not found after compilation at: ${FVECS_TO_BIN_TOOL}"
    exit 1
fi

if [ ! -f "$BASE_BIN_FILE" ]; then
    if [ -f "$BASE_FVECS_FILE" ]; then
        echo "Target file '${BASE_BIN_FILE}' not found. Starting conversion from '${BASE_FVECS_FILE}'..."
        "$FVECS_TO_BIN_TOOL" --data_type float --input_file "$BASE_FVECS_FILE" --output_file "$BASE_BIN_FILE"
        echo "Conversion complete."
    else
        echo "Warning: Source file '${BASE_FVECS_FILE}' not found. Skipping conversion. Build process might fail if base .bin is also missing."
    fi
else
    echo "Target file '${BASE_BIN_FILE}' already exists. Skipping conversion."
fi

# --- Step 3: prepare an output layout compatible with search.sh ---
if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi

echo "[INFO] Index base directory set to: $INDEX_BASE_DIR"
INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
INDEX_OUTPUT_DIR="${EXP_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"

echo "[INFO] Preparing index directory (if not exists): $INDEX_OUTPUT_DIR"
mkdir -p "$INDEX_OUTPUT_DIR/index_files"
mkdir -p "$INDEX_OUTPUT_DIR/acorn_output"
mkdir -p "$INDEX_OUTPUT_DIR/FAVOR"
mkdir -p "$INDEX_OUTPUT_DIR/navix_output"
mkdir -p "$INDEX_OUTPUT_DIR/others"
mkdir -p "$INDEX_OUTPUT_DIR/results"

BUILD_THREADS="${NUM_THREADS:-60}"
echo "[INFO] Build threads set to: $BUILD_THREADS"

UNG_MARKER_FILE="$INDEX_OUTPUT_DIR/index_files/.ung_built"
ACORN_MARKER_FILE="$INDEX_OUTPUT_DIR/acorn_output/.acorn_built"
FAVOR_MARKER_FILE="$INDEX_OUTPUT_DIR/FAVOR/favor.meta"
NAVIX_MARKER_FILE="$INDEX_OUTPUT_DIR/navix_output/.navix_built"
UNG_BUILD_STATUS="unknown"
FAVOR_BUILD_STATUS="unknown"

# --- Step 4: define build helpers ---

reset_directory_contents() {
    local target_dir="$1"
    if [[ -d "$target_dir" ]]; then
        find "$target_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
}

prepare_parallel_build_state() {
    echo "[INFO] Parallel/all mode requires a real rebuild. Cleaning existing UNG/FAVOR outputs first."
    reset_directory_contents "$INDEX_OUTPUT_DIR/index_files"
    reset_directory_contents "$INDEX_OUTPUT_DIR/results"
    reset_directory_contents "$INDEX_OUTPUT_DIR/FAVOR"
    rm -f "$UNG_MARKER_FILE" "$FAVOR_MARKER_FILE"
    rm -f "$INDEX_OUTPUT_DIR/others/ung_build_stats.csv" \
          "$INDEX_OUTPUT_DIR/others/favor_build_stats.csv" \
          "$INDEX_OUTPUT_DIR/others/parallel_build_summary.csv"
}

build_ung() {
    if [[ -f "$UNG_MARKER_FILE" && -f "$INDEX_OUTPUT_DIR/index_files/meta" ]]; then
        echo "[UNG] Marker file found ('$UNG_MARKER_FILE'). Skipping UNG build."
        UNG_BUILD_STATUS="skipped"
        return 0
    fi

    echo "[UNG] Build process started."
    "$UNG_EXECUTABLE" \
        --data_type float --dist_fn L2 --num_threads "$BUILD_THREADS" \
        --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
        --base_label_info_file "$DATA_DIR/${DATASET}_base_labels_info.log" \
        --base_label_tree_roots "$DATA_DIR/tree_roots.txt" \
        --index_path_prefix "$INDEX_OUTPUT_DIR/index_files/" \
        --result_path_prefix "$INDEX_OUTPUT_DIR/results/" \
        --scenario general --dataset "$DATASET" \
        > "$INDEX_OUTPUT_DIR/others/ung_build.log" 2>&1

    if [[ ! -f "$INDEX_OUTPUT_DIR/index_files/meta" ]]; then
        echo "[UNG] 错误: UNG 构建命令执行完成，但未生成 meta 文件: $INDEX_OUTPUT_DIR/index_files/meta"
        return 1
    fi

    echo "[UNG] Build process finished."
    UNG_BUILD_STATUS="built"
    touch "$UNG_MARKER_FILE"
}

parse_meta_value() {
    local meta_file="$1"
    local key="$2"
    awk -v k="$key" '
        {
            line=$0
            sub(/\r$/, "", line)
            pos=index(line, ":")
            if (pos == 0) {
                pos=index(line, "=")
            }
            if (pos > 0) {
                lhs=substr(line, 1, pos - 1)
                rhs=substr(line, pos + 1)
                gsub(/^[ \t]+|[ \t]+$/, "", lhs)
                gsub(/^[ \t]+|[ \t]+$/, "", rhs)
                if (lhs == k) {
                    print rhs
                    exit
                }
            }
        }
    ' "$meta_file" 2>/dev/null || true
}

print_acorn_meta_summary() {
    local tag="$1"
    local meta_file="$2"
    if [[ ! -f "$meta_file" ]]; then
        echo "[$tag] Meta file not found: $meta_file"
        return 0
    fi

    local build_time_s total_size_bytes index_only_size_bytes total_logical_memory_bytes index_only_logical_memory_bytes
    build_time_s=$(parse_meta_value "$meta_file" "total_build_time_s")
    if [[ -z "$build_time_s" ]]; then
        build_time_s=$(parse_meta_value "$meta_file" "build_time_s")
    fi
    total_size_bytes=$(parse_meta_value "$meta_file" "total_size_bytes")
    index_only_size_bytes=$(parse_meta_value "$meta_file" "index_only_size_bytes")
    total_logical_memory_bytes=$(parse_meta_value "$meta_file" "total_logical_memory_bytes")
    index_only_logical_memory_bytes=$(parse_meta_value "$meta_file" "index_only_logical_memory_bytes")

    echo "[$tag] build_time_s=${build_time_s}, total_size_bytes=${total_size_bytes}, index_only_size_bytes=${index_only_size_bytes}, total_logical_memory_bytes=${total_logical_memory_bytes}, index_only_logical_memory_bytes=${index_only_logical_memory_bytes}"
}

append_acorn_meta_csv() {
    local variant="$1"
    local meta_file="$2"
    local csv_file="$3"
    if [[ ! -f "$meta_file" ]]; then
        return 0
    fi

    local build_time_s total_size_bytes index_only_size_bytes total_logical_memory_bytes index_only_logical_memory_bytes
    build_time_s=$(parse_meta_value "$meta_file" "total_build_time_s")
    if [[ -z "$build_time_s" ]]; then
        build_time_s=$(parse_meta_value "$meta_file" "build_time_s")
    fi
    total_size_bytes=$(parse_meta_value "$meta_file" "total_size_bytes")
    index_only_size_bytes=$(parse_meta_value "$meta_file" "index_only_size_bytes")
    total_logical_memory_bytes=$(parse_meta_value "$meta_file" "total_logical_memory_bytes")
    index_only_logical_memory_bytes=$(parse_meta_value "$meta_file" "index_only_logical_memory_bytes")

    echo "${variant},${build_time_s},${total_size_bytes},${index_only_size_bytes},${total_logical_memory_bytes},${index_only_logical_memory_bytes}" >> "$csv_file"
}

get_fvecs_dim() {
    local fvecs_file="$1"
    od -An -td4 -N4 "$fvecs_file" | awk '{print $1}'
}

get_fvecs_rows() {
    local fvecs_file="$1"
    local dim
    local size_bytes
    local row_bytes
    dim=$(get_fvecs_dim "$fvecs_file")
    if [[ -z "$dim" || "$dim" -le 0 ]]; then
        return 1
    fi
    size_bytes=$(stat -c%s "$fvecs_file")
    row_bytes=$((4 + dim * 4))
    echo $((size_bytes / row_bytes))
}

write_favor_meta() {
    local meta_file="$1"
    local rows="$2"
    local dim="$3"
    local build_time_ms="$4"
    local serialized_size_bytes="$5"
    local serialized_size_mb
    serialized_size_mb=$(awk -v b="$serialized_size_bytes" 'BEGIN { printf "%.6f", b / (1024 * 1024) }')

    cat > "$meta_file" <<EOF
rows=${rows}
dim=${dim}
build_time_ms=${build_time_ms}
serialized_size_bytes=${serialized_size_bytes}
serialized_index_size_mb=${serialized_size_mb}
EOF
}

append_favor_meta_csv() {
    local meta_file="$1"
    local csv_file="$2"
    if [[ ! -f "$meta_file" ]]; then
        return 0
    fi

    local rows dim build_time_ms serialized_size_bytes serialized_index_size_mb
    rows=$(parse_meta_value "$meta_file" "rows")
    dim=$(parse_meta_value "$meta_file" "dim")
    build_time_ms=$(parse_meta_value "$meta_file" "build_time_ms")
    serialized_size_bytes=$(parse_meta_value "$meta_file" "serialized_size_bytes")
    serialized_index_size_mb=$(parse_meta_value "$meta_file" "serialized_index_size_mb")

    echo "${rows},${dim},${build_time_ms},${serialized_size_bytes},${serialized_index_size_mb}" >> "$csv_file"
}

append_ung_meta_csv() {
    local meta_file="$1"
    local csv_file="$2"
    if [[ ! -f "$meta_file" ]]; then
        return 0
    fi

    local index_time_ms index_size_mb
    index_time_ms=$(parse_meta_value "$meta_file" "index_time(ms)")
    index_size_mb=$(parse_meta_value "$meta_file" "index_size(MB)")

    echo "${index_time_ms},${index_size_mb}" >> "$csv_file"
}

write_parallel_build_summary() {
    local csv_file="$1"
    local ung_meta_file="$2"
    local favor_meta_file="$3"
    local ung_status="$4"
    local favor_status="$5"

    local ung_index_time_ms ung_index_size_mb
    local favor_build_time_ms favor_serialized_size_bytes favor_serialized_index_size_mb
    local parallel_build_time_ms parallel_build_time_s

    ung_index_time_ms=$(parse_meta_value "$ung_meta_file" "index_time(ms)")
    ung_index_size_mb=$(parse_meta_value "$ung_meta_file" "index_size(MB)")

    favor_build_time_ms=$(parse_meta_value "$favor_meta_file" "build_time_ms")
    favor_serialized_size_bytes=$(parse_meta_value "$favor_meta_file" "serialized_size_bytes")
    favor_serialized_index_size_mb=$(parse_meta_value "$favor_meta_file" "serialized_index_size_mb")
    if awk "BEGIN { exit !($ung_index_time_ms > $favor_build_time_ms) }"; then
        parallel_build_time_ms="$ung_index_time_ms"
    else
        parallel_build_time_ms="$favor_build_time_ms"
    fi
    parallel_build_time_s=$(awk -v ms="$parallel_build_time_ms" 'BEGIN { printf "%.3f", ms / 1000 }')

    {
        echo "ung_status,favor_status,parallel_build_time_ms,parallel_build_time_s,ung_index_time_ms,ung_index_size_mb,favor_build_time_ms,favor_serialized_size_bytes,favor_serialized_size_mb"
        echo "${ung_status},${favor_status},${parallel_build_time_ms},${parallel_build_time_s},${ung_index_time_ms},${ung_index_size_mb},${favor_build_time_ms},${favor_serialized_size_bytes},${favor_serialized_index_size_mb}"
    } > "$csv_file"
}

build_acorn() {
    local index_path_acorn="$INDEX_OUTPUT_DIR/acorn_output/acorn.index"
    local index_path_acorn1="$INDEX_OUTPUT_DIR/acorn_output/acorn1.index"
    local acorn_base_fvecs_path
    local acorn_base_label_path

    if [[ -f "$ACORN_MARKER_FILE" && -f "$index_path_acorn" && -f "$index_path_acorn1" ]]; then
        echo "[ACORN] Marker file found ('$ACORN_MARKER_FILE') and index files exist. Skipping build."
        return 0
    fi

    echo "[ACORN] Build process started."
    if [[ "$BUILD_MODE" == "serial" || "$BUILD_MODE" == "all" ]]; then
        acorn_base_fvecs_path="${INDEX_OUTPUT_DIR}/index_files/reordered_vecs.fvecs"
        acorn_base_label_path="${INDEX_OUTPUT_DIR}/index_files/reordered_labels.txt"
        echo "[ACORN] Using reordered base vectors: $acorn_base_fvecs_path"
    else
        acorn_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
        acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels.txt"
        if [[ "$BUILD_MODE" == "acorn_only" ]]; then
            acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels_reorder_ori.txt"
        fi
        echo "[ACORN] Using original base vectors: $acorn_base_fvecs_path"
    fi

    if [ ! -f "$acorn_base_fvecs_path" ] || [ ! -f "$acorn_base_label_path" ]; then
        echo "[ACORN] 错误: 必需的输入文件未找到."
        exit 1
    fi

    "$ACORN_EXECUTABLE" build \
        "$ACORN_N" "$ACORN_GAMMA" "$DATASET" "$ACORN_M" "$ACORN_M_BETA" \
        "$acorn_base_fvecs_path" "$acorn_base_label_path" "dummy_query_path" \
        "dummy_csv_dir" "dummy_avg_csv_dir" "dummy_dis_path" \
        60 1 true "10" \
        "$index_path_acorn" \
        "$index_path_acorn1" \
        0 \
        > "$INDEX_OUTPUT_DIR/others/acorn_build.log" 2>&1

    echo "[ACORN] Build process finished."
    touch "$ACORN_MARKER_FILE"
}

build_favor() {
    local favor_index_file="$INDEX_OUTPUT_DIR/FAVOR/favor.index"
    local favor_meta_file="$INDEX_OUTPUT_DIR/FAVOR/favor.meta"
    local favor_attribute_file="${FAVOR_ATTRIBUTE_FILE:-${DATA_DIR}/${DATASET}_favor_attribute.txt}"
    local favor_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
    local favor_rows
    local favor_dim
    local start_ms
    local end_ms
    local build_time_ms
    local serialized_size_bytes

    if [[ -f "$FAVOR_MARKER_FILE" && -f "$favor_index_file" ]]; then
        echo "[FAVOR] Marker file found ('$FAVOR_MARKER_FILE') and index file exists. Skipping build."
        FAVOR_BUILD_STATUS="skipped"
        return 0
    fi

    if [[ ! -f "$favor_base_fvecs_path" ]]; then
        echo "[FAVOR] 错误: base fvecs 文件未找到: $favor_base_fvecs_path"
        exit 1
    fi

    echo "[FAVOR] Build process started."
    start_ms=$(date +%s%3N)
    if [[ -f "$favor_attribute_file" ]]; then
        echo "[FAVOR] Using attribute file: $favor_attribute_file"
        "$FAVOR_EXECUTABLE" \
            "$favor_base_fvecs_path" \
            "$favor_attribute_file" \
            "$favor_index_file" \
            "$BUILD_THREADS" \
            > "$INDEX_OUTPUT_DIR/others/favor_build.log" 2>&1
    else
        echo "[FAVOR] Attribute file not found. Falling back to attribute-free FAVOR build."
        echo "[FAVOR] Building standalone FAVOR without attributes."
        "$FAVOR_EXECUTABLE" \
            "$favor_base_fvecs_path" \
            "$favor_index_file" \
            "$BUILD_THREADS" \
            > "$INDEX_OUTPUT_DIR/others/favor_build.log" 2>&1
    fi
    if [[ ! -f "$favor_index_file" ]]; then
        echo "[FAVOR] 错误: FAVOR 索引构建命令执行完成，但未生成索引文件: $favor_index_file"
        exit 1
    fi

    build_time_ms=$(parse_meta_value "$INDEX_OUTPUT_DIR/others/favor_build.log" "graph_build_time_ms")
    if [[ -z "$build_time_ms" ]]; then
        echo "[FAVOR] 错误: 未能从 favor_build.log 解析 graph_build_time_ms"
        exit 1
    fi

    favor_rows=$(get_fvecs_rows "$favor_base_fvecs_path")
    favor_dim=$(get_fvecs_dim "$favor_base_fvecs_path")
    serialized_size_bytes=$(stat -c%s "$favor_index_file")
    write_favor_meta "$favor_meta_file" "$favor_rows" "$favor_dim" "$build_time_ms" "$serialized_size_bytes"

    echo "[FAVOR] Build process finished."
    FAVOR_BUILD_STATUS="built"
    touch "$FAVOR_MARKER_FILE"
}

build_navix() {
    if [ -f "$NAVIX_MARKER_FILE" ]; then
        echo "[NAVIX] Marker file found ('$NAVIX_MARKER_FILE'). Skipping NaviX build."
        return 0
    fi

    echo "[NAVIX] Build process started."
    "$UNG_BUILD_DIR/apps/build_navix_index" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --index_output "$INDEX_OUTPUT_DIR/navix_output/hnsw_base.index" \
        --M "$ACORN_M" \
        --num_threads "$BUILD_THREADS" \
        > "$INDEX_OUTPUT_DIR/others/navix_build.log" 2>&1
    echo "[NAVIX] Build process finished."
    touch "$NAVIX_MARKER_FILE"
}

# --- Step 5: run tasks for the selected build mode ---
start_time_ms=$(date +%s%3N)

if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    echo "--- [Executing in PARALLEL/ALL mode] ---"
    prepare_parallel_build_state
    build_ung &
    ung_pid=$!
    build_favor &
    favor_pid=$!

    ung_status=0
    favor_status=0
    set +e
    wait "$ung_pid"
    ung_status=$?
    wait "$favor_pid"
    favor_status=$?
    set -e

    if [[ "$ung_status" -ne 0 || "$favor_status" -ne 0 ]]; then
        echo "[ERROR] Parallel build failed."
        if [[ "$ung_status" -ne 0 ]]; then
            echo "[ERROR] UNG build failed with status $ung_status. See: $INDEX_OUTPUT_DIR/others/ung_build.log"
        fi
        if [[ "$favor_status" -ne 0 ]]; then
            echo "[ERROR] FAVOR build failed with status $favor_status. See: $INDEX_OUTPUT_DIR/others/favor_build.log"
        fi
        exit 1
    fi

    if [[ "$BUILD_MODE" == "all" ]]; then
        build_acorn
    fi
elif [ "$BUILD_MODE" == "serial" ]; then
    echo "--- [Executing in SERIAL mode] ---"
    build_ung
    build_acorn
    build_favor
elif [ "$BUILD_MODE" == "ung_only" ]; then
    echo "--- [Executing in UNG_ONLY mode] ---"
    build_ung
elif [ "$BUILD_MODE" == "acorn_only" ]; then
    echo "--- [Executing in ACORN_ONLY mode] ---"
    build_acorn
elif [ "$BUILD_MODE" == "favor_only" ]; then
    echo "--- [Executing in FAVOR_ONLY mode] ---"
    build_favor
elif [ "$BUILD_MODE" == "navix_only" ]; then
    echo "--- [Executing in NAVIX_ONLY mode] ---"
    build_navix
elif [ "$BUILD_MODE" == "compile" ]; then
    echo "--- [Executing in COMPILE-ONLY mode] ---"
    echo "[INFO] 仅编译全部代码库（NaviX/ACORN/FAVOR/UNG），不构建任何索引。"
fi

end_time_ms=$(date +%s%3N)
duration_ms=$((end_time_ms - start_time_ms))
duration_s=$(awk -v ms="$duration_ms" 'BEGIN { printf "%.3f", ms / 1000 }')

echo "--- Build Summary ---"
echo "[SUCCESS] All build tasks complete. Total time: ${duration_ms} ms (${duration_s} s)."
echo "Indexes are saved in: $INDEX_OUTPUT_DIR"

UNG_META_CSV="$INDEX_OUTPUT_DIR/others/ung_build_stats.csv"
if [[ ! -f "$INDEX_OUTPUT_DIR/index_files/meta" ]]; then
    echo "[INFO] UNG meta file not found. Skipping UNG stats summary."
else
    echo "index_time_ms,index_size_mb" > "$UNG_META_CSV"
    append_ung_meta_csv "$INDEX_OUTPUT_DIR/index_files/meta" "$UNG_META_CSV"
    echo "UNG build stats saved to: $UNG_META_CSV"
fi

ACORN_META_CSV="$INDEX_OUTPUT_DIR/others/acorn_build_stats.csv"
if [[ ! -f "$INDEX_OUTPUT_DIR/acorn_output/acorn.index.meta" && ! -f "$INDEX_OUTPUT_DIR/acorn_output/acorn1.index.meta" ]]; then
    echo "[INFO] ACORN meta files not found. Skipping ACORN stats summary."
else
    echo "variant,build_time_s,total_size_bytes,index_only_size_bytes,total_logical_memory_bytes,index_only_logical_memory_bytes" > "$ACORN_META_CSV"
    print_acorn_meta_summary "ACORN-gamma" "$INDEX_OUTPUT_DIR/acorn_output/acorn.index.meta"
    append_acorn_meta_csv "acorn_gamma" "$INDEX_OUTPUT_DIR/acorn_output/acorn.index.meta" "$ACORN_META_CSV"
    if [[ -f "$INDEX_OUTPUT_DIR/acorn_output/acorn1.index.meta" ]]; then
        print_acorn_meta_summary "ACORN-1" "$INDEX_OUTPUT_DIR/acorn_output/acorn1.index.meta"
        append_acorn_meta_csv "acorn_1" "$INDEX_OUTPUT_DIR/acorn_output/acorn1.index.meta" "$ACORN_META_CSV"
    fi
    echo "ACORN build stats saved to: $ACORN_META_CSV"
fi

FAVOR_META_CSV="$INDEX_OUTPUT_DIR/others/favor_build_stats.csv"
if [[ ! -f "$INDEX_OUTPUT_DIR/FAVOR/favor.meta" ]]; then
    echo "[INFO] FAVOR meta file not found. Skipping FAVOR stats summary."
else
    echo "rows,dim,build_time_ms,serialized_size_bytes,serialized_index_size_mb" > "$FAVOR_META_CSV"
    append_favor_meta_csv "$INDEX_OUTPUT_DIR/FAVOR/favor.meta" "$FAVOR_META_CSV"
    echo "FAVOR build stats saved to: $FAVOR_META_CSV"
fi

PARALLEL_SUMMARY_CSV="$INDEX_OUTPUT_DIR/others/parallel_build_summary.csv"
if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    if [[ -f "$INDEX_OUTPUT_DIR/index_files/meta" && -f "$INDEX_OUTPUT_DIR/FAVOR/favor.meta" ]]; then
        if [[ "$UNG_BUILD_STATUS" == "skipped" && "$FAVOR_BUILD_STATUS" == "skipped" && -f "$PARALLEL_SUMMARY_CSV" ]]; then
            echo "[INFO] UNG and FAVOR were both skipped. Keeping existing parallel summary: $PARALLEL_SUMMARY_CSV"
        else
            write_parallel_build_summary "$PARALLEL_SUMMARY_CSV" "$INDEX_OUTPUT_DIR/index_files/meta" "$INDEX_OUTPUT_DIR/FAVOR/favor.meta" "$UNG_BUILD_STATUS" "$FAVOR_BUILD_STATUS"
            echo "Parallel build summary saved to: $PARALLEL_SUMMARY_CSV"
        fi
    else
        echo "[INFO] Missing UNG or FAVOR meta file. Skipping parallel build summary."
    fi
fi
