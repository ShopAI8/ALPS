#!/bin/bash

# ==============================================================================
# search.sh - Execute search workloads using prebuilt indexes and ground-truth data
# ==============================================================================

set -e # Exit immediately if any command fails
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Step 1: Parse command-line arguments ---
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        if [[ $key == "QUERY_DIR_NAME" ]]; then
            QUERY_DIR_NAME="$2"
            shift 2
            continue
        fi
        if [[ $key == "ROUTING_MODE" ]]; then
            ROUTING_MODE="$2"
            shift 2
            continue
        fi
        if [[ $key == "BASELINE_ALG" ]]; then
            BASELINE_ALG="$2"
            shift 2
            continue
        fi
        if [ -z "$2" ]; then
            echo "Error: Missing value for parameter $1"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "Unknown parameter: $1"; exit 1
    fi
done

# Default parameters
if [ -z "$UNG_DISTANCE_MODE" ]; then
    UNG_DISTANCE_MODE="exact"
fi
if [[ "$UNG_DISTANCE_MODE" == "rabitq" ]]; then
    echo "[WARN] RabitQ support is disabled. Falling back to exact UNG distance mode."
    UNG_DISTANCE_MODE="exact"
fi

# --- Step 2: Construct a unique output directory based on the search parameters ---
SAFE_QUERY_NAME=$(echo "$QUERY_DIR_NAME" | tr '/' '_')
GT_DIR_NAME="GT_${SAFE_QUERY_NAME}_K${K}"
SEARCH_DIR_NAME="Ls${LSEARCH_START}-Le${LSEARCH_END}-Lp${LSEARCH_STEP}_efsS${EFS_START}-efss${EFS_STEP_SLOW}-efsf${EFS_STEP_FAST}-lt${LSEARCH_THRESHOLD}_K${K}_th${NUM_THREADS}"
RESULT_OUTPUT_DIR="${ALGO_RESULT_DIR}/Index[${INDEX_DIR_NAME}]_GT[${GT_DIR_NAME}]_Search[${SEARCH_DIR_NAME}]"

# --- Step 3: Create result directories ---
mkdir -p "$RESULT_OUTPUT_DIR/results"
mkdir -p "$RESULT_OUTPUT_DIR/others"
mkdir -p "$(dirname "$CURATOR_INDEX_PATH")"

# --- Step 4: Prepare the Lsearch parameter sequence ---
LSEARCH_VALUES=$(seq "$LSEARCH_START" "$LSEARCH_STEP" "$LSEARCH_END" | tr '\n' ' ')
echo "Evaluating the following Lsearch values: $LSEARCH_VALUES"

# --- Step 5: Define dependent file and directory paths ---
# Select the index base directory according to the build mode
if [[ "$BUILD_MODE" == "parallel" ]]; then
    INDEX_BASE_DIR="Index_parallel"
elif [[ "$BUILD_MODE" == "skip" ]]; then
    # Skip mode: the index was built by a previous run, whose base dir is unknown.
    # Prefer the parallel dir if this exact index was built there, else fall back to Index.
    if [[ -d "${SHARED_OUTPUT_DIR}/Index_parallel/${INDEX_DIR_NAME}" ]]; then
        INDEX_BASE_DIR="Index_parallel"
        echo "[INFO] Skip mode: parallel index found at Index_parallel/${INDEX_DIR_NAME}."
    else
        INDEX_BASE_DIR="Index"
        echo "[INFO] Skip mode: no parallel index found, using Index/${INDEX_DIR_NAME}."
    fi
else
    INDEX_BASE_DIR="Index"
fi
INDEX_PATH="${SHARED_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"
GT_PATH="${SHARED_OUTPUT_DIR}/GroundTruth/${GT_DIR_NAME}"
MODEL_PATH="${SHARED_OUTPUT_DIR}/SelectModels"
if [[ -n "${SELECTOR_MODEL_PATH:-}" ]]; then
    MODEL_PATH="$SELECTOR_MODEL_PATH"
fi
CURATOR_INDEX_PATH="${SHARED_OUTPUT_DIR}/Index/CuratorIndex/curator_index.bin"
# MODEL_PATH="/noraiddata/lijiakang/FilterVector/FilterVectorResults/OLD/${DATASET}/SelectModels"
ACORN_INDEX_PREFIX="${INDEX_PATH}/acorn_output"
NAVIX_INDEX_PATH="${INDEX_PATH}/navix_output/hnsw_base.index"
ACORN_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn.index"
ACORN_1_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn1.index"

QUERY_DIR="${DATA_DIR}/${QUERY_DIR_NAME}"
echo "Using query directory from: $QUERY_DIR"

# In SmartRoute++ / RabitQ-based scenarios, UNG uses the RQB index.
# If the RQB directory does not contain the required ACORN or NaviX files,
# automatically fall back to the corresponding artifacts in the non-RQB directory.
if [[ "$UNG_DISTANCE_MODE" == "rabitq" && "$INDEX_DIR_NAME" =~ ^(.*)_RQB[0-9]+$ ]]; then
    EXACT_INDEX_DIR_NAME="${BASH_REMATCH[1]}"
    EXACT_INDEX_PATH="${SHARED_OUTPUT_DIR}/${INDEX_BASE_DIR}/${EXACT_INDEX_DIR_NAME}"

    # In RabitQ mode, prefer RabitQ ACORN files; if unavailable in the RQB
    # directory, fall back to the standard ACORN files in the exact directory.
    if [[ ! -f "${ACORN_INDEX_PREFIX}/acorn_rabitq.index" || ! -f "${ACORN_INDEX_PREFIX}/acorn1_rabitq.index" ]]; then
        if [[ -f "${EXACT_INDEX_PATH}/acorn_output/acorn.index" && -f "${EXACT_INDEX_PATH}/acorn_output/acorn1.index" ]]; then
            echo "[INFO] ACORN RAbitQ index files not found in RQB dir, fallback to exact dir: ${EXACT_INDEX_PATH}/acorn_output"
            ACORN_INDEX_PREFIX="${EXACT_INDEX_PATH}/acorn_output"
        fi
    fi

    if [[ ! -f "$NAVIX_INDEX_PATH" && -f "${EXACT_INDEX_PATH}/navix_output/hnsw_base.index" ]]; then
        echo "[INFO] NaviX index not found in RQB dir, fallback to exact dir: ${EXACT_INDEX_PATH}/navix_output/hnsw_base.index"
        NAVIX_INDEX_PATH="${EXACT_INDEX_PATH}/navix_output/hnsw_base.index"
    fi
fi

# Reset the default index paths based on the final ACORN_INDEX_PREFIX
ACORN_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn.index"
ACORN_1_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn1.index"

if [[ "$UNG_DISTANCE_MODE" == "rabitq" ]]; then
    if [[ -f "${ACORN_INDEX_PREFIX}/acorn_rabitq.index" && -f "${ACORN_INDEX_PREFIX}/acorn1_rabitq.index" ]]; then
        ACORN_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn_rabitq.index"
        ACORN_1_INDEX_FILE="${ACORN_INDEX_PREFIX}/acorn1_rabitq.index"
        echo "[INFO] Using ACORN RAbitQ index files."
    else
        echo "[INFO] ACORN RAbitQ index files not found, fallback to standard ACORN files."
    fi
fi

echo "Using index directory: $INDEX_PATH"
echo "Using ground-truth directory: $GT_PATH"
echo "Search results will be written to: $RESULT_OUTPUT_DIR"
echo "Using ACORN index: ${ACORN_INDEX_FILE}"
echo "Using NaviX index: $NAVIX_INDEX_PATH"

# --- Step 6: Ensure query is in .bin format ---
QUERY_FVECS="${QUERY_DIR}/${DATASET}_query.fvecs"
QUERY_BIN="${QUERY_DIR}/${DATASET}_query.bin"
if [ ! -f "$QUERY_BIN" ] && [ -f "$QUERY_FVECS" ]; then
    echo "Converting query from .fvecs to .bin..."
    FVECS_TO_BIN="$BUILD_DIR/tools/fvecs_to_bin"
    if [ ! -x "$FVECS_TO_BIN" ]; then
        # Try Python fallback
        python3 -c "
import struct, sys
with open('$QUERY_FVECS','rb') as f: data = f.read()
n, pos, vecs = 0, 0, []
while pos < len(data):
    d = struct.unpack_from('i', data, pos)[0]; pos += 4
    vecs.append(data[pos:pos+d*4]); pos += d*4; n += 1
with open('$QUERY_BIN','wb') as out:
    out.write(struct.pack('II', n, d))
    for v in vecs: out.write(v)
print(f'Converted {n} queries x {d} dim')
"
    else
        "$FVECS_TO_BIN" --data_type float --input_file "$QUERY_FVECS" --output_file "$QUERY_BIN"
    fi
fi

# --- Step 7: Execute the search workload ---
PERF_EVENTS="cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss,LLC-loads,LLC-load-misses,branches,branch-misses"
PERF_LOG_PATH="$RESULT_OUTPUT_DIR/others/${DATASET}_perf_stat.log"
SEARCH_LOG_PATH="$RESULT_OUTPUT_DIR/others/${DATASET}_search_output.txt"
read -r -a LSEARCH_ARGS <<< "$LSEARCH_VALUES"
SEARCH_COMMAND=(
  "$BUILD_DIR/apps/search_UNG_index"
    --data_type float  --dataset "$DATASET" --dist_fn L2 --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
    --is_new_method true \
    --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
    --routing_mode "$ROUTING_MODE" \
    --baseline_alg "$BASELINE_ALG" \
    --curator_nlist "${CURATOR_NLIST:-32}" \
    --curator_nprobe "${CURATOR_NPROBE:-16}" \
    --curator_max_leaf_size "${CURATOR_MAX_LEAF_SIZE:-128}" \
    --curator_search_ef "${CURATOR_SEARCH_EF:-10}" \
    --curator_beam_size "${CURATOR_BEAM_SIZE:-1}" \
    --curator_index_path "$CURATOR_INDEX_PATH" \
    --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
    --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
    --query_bin_file "$QUERY_DIR/${DATASET}_query.bin" \
    --query_label_file "$QUERY_DIR/${DATASET}_query_labels.txt" \
    --query_group_id_file "$QUERY_DIR/${DATASET}_query_source_groups.txt" \
    --gt_file "$GT_PATH/${DATASET}_gt_labels_containment.bin" \
    --index_path_prefix "$INDEX_PATH/index_files/" \
    --result_path_prefix "$RESULT_OUTPUT_DIR/results/" \
    --acorn_index_path "${ACORN_INDEX_FILE}" --acorn_1_index_path "${ACORN_1_INDEX_FILE}" \
    --selector_modle_prefix "${MODEL_PATH}" \
    --scenario containment \
    --num_entry_points "$NUM_ENTRY_POINTS" \
    --Lsearch "${LSEARCH_ARGS[@]}" \
    --lsearch_start "$LSEARCH_START" \
    --lsearch_step "$LSEARCH_STEP" \
    --efs_start "$EFS_START" \
    --efs_step_slow "$EFS_STEP_SLOW" --efs_step_fast "$EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD" \
    --ung_distance_mode "$UNG_DISTANCE_MODE" \
    --navix_index_path "$NAVIX_INDEX_PATH" \
    --algo_choice_csv "${ALGO_CHOICE_CSV:-$QUERY_DIR/algo_choice_repeat.csv}" \
    --optimize_standalone_prefilter "${OPTIMIZE_STANDALONE_PREFILTER:-false}"
)

# Hardware counters are normally unavailable in an unprivileged container.
# Use them when possible, otherwise run the exact same search command directly.
if [[ "${ALPS_ENABLE_PERF:-auto}" != "0" ]] \
   && command -v perf >/dev/null 2>&1 \
   && perf stat -e task-clock true >/dev/null 2>&1; then
    echo "Performance profiling output (perf stat) will be saved to: $PERF_LOG_PATH"
    perf stat -e "$PERF_EVENTS" -o "$PERF_LOG_PATH" \
        "${SEARCH_COMMAND[@]}" > "$SEARCH_LOG_PATH" 2>&1
else
    echo "[INFO] perf is disabled or unavailable; running search without hardware counters."
    "${SEARCH_COMMAND[@]}" > "$SEARCH_LOG_PATH" 2>&1
fi

# --- Step 7: Post-process results and compute global averages ---
echo "Computing global averages across all query-level metrics..."
DETAILS_CSV="${RESULT_OUTPUT_DIR}/results/query_details_repeat${NUM_REPEATS}.csv"
AVERAGE_CSV="${RESULT_OUTPUT_DIR}/results/query_details_global_average.csv"

if [ -f "$DETAILS_CSV" ]; then
    python3 UNG/data/average_query_details.py --input_csv "$DETAILS_CSV" --output_csv "$AVERAGE_CSV"
else
    echo "Warning: Detail file $DETAILS_CSV was not found. Skipping global average computation."
fi


echo "All search and post-processing tasks have completed successfully."
