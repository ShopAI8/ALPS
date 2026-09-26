#!/bin/bash

# Compile and build the indexes required by ALPS, ALPS+, and TFNG.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CMAKE="${CMAKE:-cmake}"

while [[ $# -gt 0 ]]; do
    if [[ "$1" != --* || $# -lt 2 ]]; then
        echo "错误: 参数 '$1' 缺少值或格式无效"
        exit 1
    fi
    key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
    declare "$key"="$2"
    shift 2
done

: "${BUILD_MODE:?缺少 --build_mode}"

case "$BUILD_MODE" in
    parallel|serial|all|ung_only|favor_only|skip|compile) ;;
    *)
        echo "错误: 无效的 build_mode '$BUILD_MODE'。可用选项: parallel, serial, all, ung_only, favor_only, skip, compile"
        exit 1
        ;;
esac

if [[ "$BUILD_MODE" == "skip" ]]; then
    echo "[INFO] Build mode is 'skip'; using existing ALPS indexes."
    exit 0
fi

UNG_BUILD_DIR="${UNG_BUILD_DIR:-${SCRIPT_DIR}/UNG/codes/build}"
FAVOR_BUILD_DIR="${FAVOR_BUILD_DIR:-${SCRIPT_DIR}/FAVOR/build}"

echo "[INFO] Compiling UNG/TFNG core (legacy baselines are not part of this build)."
mkdir -p "$UNG_BUILD_DIR"
"$CMAKE" -S "${SCRIPT_DIR}/UNG/codes" -B "$UNG_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"$CMAKE" --build "$UNG_BUILD_DIR" --parallel "$BUILD_JOBS"

echo "[INFO] Compiling FAVOR, the high-selectivity execution path used by ALPS."
mkdir -p "$FAVOR_BUILD_DIR"
"$CMAKE" -S "${SCRIPT_DIR}/FAVOR" -B "$FAVOR_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"$CMAKE" --build "$FAVOR_BUILD_DIR" --target build_index --parallel "$BUILD_JOBS"

if [[ "$BUILD_MODE" == "compile" ]]; then
    echo "[SUCCESS] ALPS, ALPS+, and TFNG components compiled successfully."
    exit 0
fi

: "${DATASET:?缺少 --dataset}"
: "${DATA_DIR:?缺少 --data_dir}"
: "${EXP_OUTPUT_DIR:?缺少 --exp_output_dir}"
: "${MAX_DEGREE:?缺少 --max_degree}"
: "${LBUILD:?缺少 --Lbuild}"
: "${ALPHA:?缺少 --alpha}"
: "${NUM_CROSS_EDGES:?缺少 --num_cross_edges}"
: "${NUM_ENTRY_POINTS:?缺少 --num_entry_points}"

FVECS_TO_BIN_TOOL="${UNG_BUILD_DIR}/tools/fvecs_to_bin"
UNG_EXECUTABLE="${UNG_BUILD_DIR}/apps/build_UNG_index"
FAVOR_EXECUTABLE="${FAVOR_BUILD_DIR}/app/build_index"
BASE_FVECS_FILE="${DATA_DIR}/${DATASET}_base.fvecs"
BASE_BIN_FILE="${DATA_DIR}/${DATASET}_base.bin"

for executable in "$FVECS_TO_BIN_TOOL" "$UNG_EXECUTABLE" "$FAVOR_EXECUTABLE"; do
    if [[ ! -x "$executable" ]]; then
        echo "错误: 编译产物不存在或不可执行: $executable"
        exit 1
    fi
done

if [[ ! -f "$BASE_BIN_FILE" ]]; then
    if [[ ! -f "$BASE_FVECS_FILE" ]]; then
        echo "错误: 缺少底库向量文件: $BASE_BIN_FILE 或 $BASE_FVECS_FILE"
        exit 1
    fi
    "$FVECS_TO_BIN_TOOL" --data_type float --input_file "$BASE_FVECS_FILE" --output_file "$BASE_BIN_FILE"
fi

if [[ "$BUILD_MODE" == "parallel" ]]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi

INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}"
if [[ "${BUILD_RABITQ_SIDE_INDEX:-false}" == "true" ]]; then
    INDEX_DIR_NAME="${INDEX_DIR_NAME}_RQB${RABITQ_TOTAL_BITS:-4}"
fi
INDEX_OUTPUT_DIR="${EXP_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"
mkdir -p "$INDEX_OUTPUT_DIR/index_files" "$INDEX_OUTPUT_DIR/FAVOR" "$INDEX_OUTPUT_DIR/others" "$INDEX_OUTPUT_DIR/results"

BUILD_THREADS="${NUM_THREADS:-60}"

build_ung() {
    local marker="$INDEX_OUTPUT_DIR/index_files/.ung_built"
    if [[ -f "$marker" && -f "$INDEX_OUTPUT_DIR/index_files/meta" ]]; then
        echo "[UNG] Existing index found; skipping."
        return
    fi

    "$UNG_EXECUTABLE" \
        --data_type float --dist_fn L2 --num_threads "$BUILD_THREADS" \
        --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
        --num_cross_edges "$NUM_CROSS_EDGES" \
        --base_bin_file "$BASE_BIN_FILE" \
        --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
        --base_label_info_file "$DATA_DIR/${DATASET}_base_labels_info.log" \
        --base_label_tree_roots "$DATA_DIR/tree_roots.txt" \
        --index_path_prefix "$INDEX_OUTPUT_DIR/index_files/" \
        --result_path_prefix "$INDEX_OUTPUT_DIR/results/" \
        --scenario general --dataset "$DATASET" \
        --build_rabitq_side_index "${BUILD_RABITQ_SIDE_INDEX:-false}" \
        --rabitq_total_bits "${RABITQ_TOTAL_BITS:-4}" \
        >"$INDEX_OUTPUT_DIR/others/ung_build.log" 2>&1

    if [[ ! -f "$INDEX_OUTPUT_DIR/index_files/meta" ]]; then
        echo "[UNG] 构建失败，未生成 meta；日志: $INDEX_OUTPUT_DIR/others/ung_build.log"
        return 1
    fi
    touch "$marker"
}

read_fvecs_dim() {
    od -An -td4 -N4 "$1" | awk '{print $1}'
}

build_favor() {
    local index_file="$INDEX_OUTPUT_DIR/FAVOR/favor.index"
    local meta_file="$INDEX_OUTPUT_DIR/FAVOR/favor.meta"
    local attribute_file="${FAVOR_ATTRIBUTE_FILE:-${DATA_DIR}/${DATASET}_favor_attribute.txt}"
    if [[ -f "$meta_file" && -f "$index_file" ]]; then
        echo "[FAVOR] Existing index found; skipping."
        return
    fi
    if [[ ! -f "$BASE_FVECS_FILE" ]]; then
        echo "[FAVOR] 缺少底库 fvecs 文件: $BASE_FVECS_FILE"
        return 1
    fi

    local start_ms end_ms size_bytes dim rows row_bytes file_bytes
    start_ms=$(date +%s%3N)
    if [[ -f "$attribute_file" ]]; then
        "$FAVOR_EXECUTABLE" "$BASE_FVECS_FILE" "$attribute_file" "$index_file" "$BUILD_THREADS" \
            >"$INDEX_OUTPUT_DIR/others/favor_build.log" 2>&1
    else
        "$FAVOR_EXECUTABLE" "$BASE_FVECS_FILE" "$index_file" "$BUILD_THREADS" \
            >"$INDEX_OUTPUT_DIR/others/favor_build.log" 2>&1
    fi
    end_ms=$(date +%s%3N)
    size_bytes=$(stat -c%s "$index_file")
    dim=$(read_fvecs_dim "$BASE_FVECS_FILE")
    row_bytes=$((4 + dim * 4))
    file_bytes=$(stat -c%s "$BASE_FVECS_FILE")
    rows=$((file_bytes / row_bytes))
    printf 'rows=%s\ndim=%s\nbuild_time_ms=%s\nserialized_size_bytes=%s\n' \
        "$rows" "$dim" "$((end_ms - start_ms))" "$size_bytes" >"$meta_file"
}

case "$BUILD_MODE" in
    parallel)
        build_ung & ung_pid=$!
        build_favor & favor_pid=$!
        wait "$ung_pid"
        wait "$favor_pid"
        ;;
    serial|all)
        build_ung
        build_favor
        ;;
    ung_only) build_ung ;;
    favor_only) build_favor ;;
esac

echo "[SUCCESS] Required indexes are available in: $INDEX_OUTPUT_DIR"
