#!/bin/bash

set -e # Exit immediately if any command fails.

IS_NEW_TRIE_METHOD=false

# --- Resolve script and config paths ---
if [ -z "$1" ]; then
    echo "错误: 请提供一个 JSON 配置文件作为第一个参数。"
    echo "用法: ./exp.sh [config_file.json]"
    exit 1
fi

CONFIG_FILE="$1"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件未找到，请检查路径: $CONFIG_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装 jq (https://stedolan.github.io/jq/)"
    exit 1
fi

if [[ -n "${ALPS_ALGORITHMS:-}" ]]; then
    IFS=',' read -r -a requested_algorithms <<< "$ALPS_ALGORITHMS"
    for requested_algorithm in "${requested_algorithms[@]}"; do
        case "$requested_algorithm" in
            ALPS|ALPS+|TFNG) ;;
            *)
                echo "错误: 不支持算法 '$requested_algorithm'。当前仅支持 ALPS、ALPS+ 和 TFNG。"
                exit 1
                ;;
        esac
    done
fi

echo "成功找到配置文件: $CONFIG_FILE"
echo "开始执行实验..."

while read -r dataset_config; do
    
    # --- Step A: read shared parameters as defaults ---
    DATASET=$(echo "$dataset_config" | jq -r '.dataset_name')
    SHARED_CONFIG=$(echo "$dataset_config" | jq '.shared_config')
    
    DATA_DIR=$(echo "$SHARED_CONFIG" | jq -r '.data_dir')
    BASE_OUTPUT_DIR=$(echo "$SHARED_CONFIG" | jq -r '.output_dir')
    # Container-friendly overrides. The JSON files can keep the original host
    # paths while Docker mounts datasets at /data and outputs at /results.
    if [[ -n "${ALPS_DATA_ROOT:-}" ]]; then
        DATA_DIR="${ALPS_DATA_ROOT%/}/${DATASET}"
    fi
    if [[ -n "${ALPS_OUTPUT_ROOT:-}" ]]; then
        BASE_OUTPUT_DIR="${ALPS_OUTPUT_ROOT%/}"
    fi
    BUILD_MODE="${ALPS_BUILD_MODE:-$(echo "$SHARED_CONFIG" | jq -r '.build_mode')}"
    MAX_DEGREE=$(echo "$SHARED_CONFIG" | jq -r '.max_degree')
    LBUILD=$(echo "$SHARED_CONFIG" | jq -r '.Lbuild')
    ALPHA=$(echo "$SHARED_CONFIG" | jq -r '.alpha')
    NUM_CROSS_EDGES=$(echo "$SHARED_CONFIG" | jq -r '.num_cross_edges')
    NUM_ENTRY_POINTS=$(echo "$SHARED_CONFIG" | jq -r '.num_entry_points')
    K=$(echo "$SHARED_CONFIG" | jq -r '.K')
    LSEARCH_START=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_start')
    LSEARCH_END=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_end')
    LSEARCH_STEP=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_step')
    NUM_THREADS=$(echo "$SHARED_CONFIG" | jq -r '.num_threads')
    NUM_REPEATS=$(echo "$SHARED_CONFIG" | jq -r '.num_repeats')
    # ALPS switches between the slow and fast FAVOR-HNSW ef increments at this
    # Lsearch threshold.
    LSEARCH_THRESHOLD=$(echo "$SHARED_CONFIG" | jq -r '.alps_params.efs_threshold')
    BUILD_RABITQ_SIDE_INDEX=$(echo "$SHARED_CONFIG" | jq -r '.build_rabitq_side_index // false')
    RABITQ_TOTAL_BITS=$(echo "$SHARED_CONFIG" | jq -r '.rabitq_total_bits // 4')
    UNG_DISTANCE_MODE_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.ung_distance_mode // "exact"')
    OPTIMIZE_STANDALONE_PREFILTER=$(echo "$SHARED_CONFIG" | jq -r '.optimize_standalone_prefilter // false')
    SELECTOR_MODEL_PATH_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.selector_model_path // ""')
    # The C++ router maps the historical class label FAVOR to the ef-aligned
    # FAVOR-HNSW path.  Keep this optional override empty by default so result
    # directories remain Results/ALPS and Results/ALPS+.
    ROUTER_ZERO_ALGORITHM_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.router_zero_algorithm // ""')
    RESULT_NAME_SUFFIX_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.result_name_suffix // ""')
    ALGO_CHOICE_CSV_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.algo_choice_csv // ""')

    # Resolve the project root from PROJECT_ROOT, or infer it from this script.
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
    if [[ -n "${ALPS_BUILD_ROOT:-}" ]]; then
        export UNG_BUILD_DIR="${ALPS_BUILD_ROOT%/}/ung"
        export FAVOR_BUILD_DIR="${ALPS_BUILD_ROOT%/}/favor"
    else
        export UNG_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/ung"
        export FAVOR_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/favor"
    fi
    BUILD_ONLY_INDEX_PREPARED=false

    # --- Middle loop: iterate over query tasks ---
    while read -r task; do
        QUERY_DIR_NAME=$(echo "$task" | jq -r '.query_dir_name')

        # --- Load and override task-specific parameters ---
        EFS_START=$(echo "$task" | jq -r '.alps_search_params.efs_start')
        EFS_STEP_SLOW=$(echo "$task" | jq -r '.alps_search_params.efs_step_slow')
        EFS_STEP_FAST=$(echo "$task" | jq -r '.alps_search_params.efs_step_fast')
        TASK_SELECTOR_MODEL_PATH=$(echo "$task" | jq -r '.selector_model_path // ""')
        TASK_ROUTER_ZERO_ALGORITHM=$(echo "$task" | jq -r '.router_zero_algorithm // ""')
        TASK_RESULT_NAME_SUFFIX=$(echo "$task" | jq -r '.result_name_suffix // ""')
        TASK_ALGO_CHOICE_CSV=$(echo "$task" | jq -r '.algo_choice_csv // ""')

        EFFECTIVE_SELECTOR_MODEL_PATH="$SELECTOR_MODEL_PATH_DEFAULT"
        if [[ -n "$TASK_SELECTOR_MODEL_PATH" && "$TASK_SELECTOR_MODEL_PATH" != "null" ]]; then
            EFFECTIVE_SELECTOR_MODEL_PATH="$TASK_SELECTOR_MODEL_PATH"
        fi

        EFFECTIVE_ROUTER_ZERO_ALGORITHM="$ROUTER_ZERO_ALGORITHM_DEFAULT"
        if [[ -n "$TASK_ROUTER_ZERO_ALGORITHM" && "$TASK_ROUTER_ZERO_ALGORITHM" != "null" ]]; then
            EFFECTIVE_ROUTER_ZERO_ALGORITHM="$TASK_ROUTER_ZERO_ALGORITHM"
        fi

        EFFECTIVE_RESULT_NAME_SUFFIX="$RESULT_NAME_SUFFIX_DEFAULT"
        if [[ -n "$TASK_RESULT_NAME_SUFFIX" && "$TASK_RESULT_NAME_SUFFIX" != "null" ]]; then
            EFFECTIVE_RESULT_NAME_SUFFIX="$TASK_RESULT_NAME_SUFFIX"
        fi

        EFFECTIVE_ALGO_CHOICE_CSV="$ALGO_CHOICE_CSV_DEFAULT"
        if [[ -n "$TASK_ALGO_CHOICE_CSV" && "$TASK_ALGO_CHOICE_CSV" != "null" ]]; then
            EFFECTIVE_ALGO_CHOICE_CSV="$TASK_ALGO_CHOICE_CSV"
        fi

        # Fail fast if the ef sweep required by ALPS' FAVOR-HNSW path is missing.
        if [[ "$EFS_START" == "null" || -z "$EFS_START" ]]; then
            echo "错误: 任务 '$QUERY_DIR_NAME' 缺少 'alps_search_params.efs_start' 参数！"
            exit 1
        fi

        # --- Inner loop: iterate over algorithm names ---
        while read -r ALGORITHM_NAME; do
            
            echo -e "\n=========================================================="
            echo "Processing: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME]"
            echo "Using ALPS ef sweep: start=${EFS_START}, slow_step=${EFS_STEP_SLOW}, fast_step=${EFS_STEP_FAST}"
            echo "=========================================================="

            # Map the algorithm name to its runtime parameters.
            case "$ALGORITHM_NAME" in
                "TFNG") ROUTING_MODE=0; BASELINE_ALG=15; IS_REC_MORE_START=false;;
                "ALPS")     ROUTING_MODE=1; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                "ALPS+")    ROUTING_MODE=5; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                *)
                    echo "错误: 未知的算法名称 '$ALGORITHM_NAME'。请在 exp.sh 的 case 语句中定义它。"
                    exit 1;;
            esac

            # Follow the JSON config by default, with targeted overrides when needed.
            UNG_DISTANCE_MODE="$UNG_DISTANCE_MODE_DEFAULT"
            # Use the JSON-configured RabitQ side-index setting.
            EFFECTIVE_BUILD_RABITQ_SIDE_INDEX="$BUILD_RABITQ_SIDE_INDEX"
            
            SHARED_DATASET_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
            RESULT_ALGORITHM_NAME="${ALGORITHM_NAME}"
            if [[ -n "$EFFECTIVE_RESULT_NAME_SUFFIX" ]]; then
                RESULT_ALGORITHM_NAME="${ALGORITHM_NAME}_${EFFECTIVE_RESULT_NAME_SUFFIX}"
            elif [[ "$ROUTING_MODE" -ne 0 && -n "$EFFECTIVE_ROUTER_ZERO_ALGORITHM" ]]; then
                RESULT_ALGORITHM_NAME="${ALGORITHM_NAME}_${EFFECTIVE_ROUTER_ZERO_ALGORITHM}"
            fi
            ALGO_RESULT_DIR="${SHARED_DATASET_DIR}/Results/${RESULT_ALGORITHM_NAME}"

            # --- Call build_hybrid.sh ---
            # build_hybrid.sh handles compilation, data conversion, and index building.
            # `skip` mode bypasses the build entirely and goes straight to query.
            if [[ "$BUILD_MODE" == "skip" ]]; then
                echo "[INFO] Build mode is 'skip'. Skipping index build. Proceeding directly to ground truth + search."
            elif [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "ung_only" || "$BUILD_MODE" == "favor_only" ]]; then
                if [[ "$BUILD_ONLY_INDEX_PREPARED" == true ]]; then
                    echo "Preparing build index..."
                    echo "[INFO] Build-only mode: index already prepared for dataset '$DATASET' in this run. Skipping duplicate rebuild."
                else
                    echo "Preparing build index..."
                    ./build_hybrid.sh \
                       --build_mode "$BUILD_MODE" \
                       --query_dir_name "$QUERY_DIR_NAME" \
                       --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
                       --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
                       --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
                       --build_rabitq_side_index "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" \
                       --rabitq_total_bits "$RABITQ_TOTAL_BITS"
                    BUILD_ONLY_INDEX_PREPARED=true
                fi
            else
                echo "Preparing build index..."
                ./build_hybrid.sh \
                   --build_mode "$BUILD_MODE" \
                   --query_dir_name "$QUERY_DIR_NAME" \
                   --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
                   --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
                   --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
                   --build_rabitq_side_index "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" \
                   --rabitq_total_bits "$RABITQ_TOTAL_BITS"
            fi
            
            # Some build modes are build-only and should skip GT generation and search.
            if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "ung_only" || "$BUILD_MODE" == "favor_only" || "$BUILD_MODE" == "compile" ]]; then
               echo "[INFO] Skipping GT generation and search steps."
               echo "--- The current experimental configuration processing has been completed (BUILD ONLY) ---"
               continue
            fi
            
            # --- Call generate_gt.sh ---
            echo "Preparing Ground Truth (K=$K)..."
            ./generate_gt.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" --build_dir "$UNG_BUILD_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --K "$K"

            # --- Call search.sh ---
            INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}"
            if [[ "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" == "true" ]]; then
               INDEX_DIR_NAME="${INDEX_DIR_NAME}_RQB${RABITQ_TOTAL_BITS}"
            fi
            echo "Using UNG distance mode: $UNG_DISTANCE_MODE"
            echo "Using RabitQ side index: $EFFECTIVE_BUILD_RABITQ_SIDE_INDEX"
            echo "Using index dir name: $INDEX_DIR_NAME"
            if [[ -n "$EFFECTIVE_SELECTOR_MODEL_PATH" ]]; then
               echo "Using selector model path override: $EFFECTIVE_SELECTOR_MODEL_PATH"
            fi
            if [[ -n "$EFFECTIVE_ROUTER_ZERO_ALGORITHM" ]]; then
               echo "Using ALPS class-0 override: $EFFECTIVE_ROUTER_ZERO_ALGORITHM"
            fi
            if [[ -n "$EFFECTIVE_RESULT_NAME_SUFFIX" ]]; then
               echo "Using result name suffix override: $EFFECTIVE_RESULT_NAME_SUFFIX"
            fi
            if [[ -n "$EFFECTIVE_ALGO_CHOICE_CSV" ]]; then
               echo "Using per-query oracle choices: $EFFECTIVE_ALGO_CHOICE_CSV"
            fi
            echo "Begin search (K=$K)..."
            if [[ -n "$EFFECTIVE_SELECTOR_MODEL_PATH" ]]; then
               export SELECTOR_MODEL_PATH="$EFFECTIVE_SELECTOR_MODEL_PATH"
            else
               unset SELECTOR_MODEL_PATH
            fi
            if [[ -n "$EFFECTIVE_ROUTER_ZERO_ALGORITHM" ]]; then
               export ROUTER_ZERO_ALGORITHM="$EFFECTIVE_ROUTER_ZERO_ALGORITHM"
            else
               unset ROUTER_ZERO_ALGORITHM
            fi

            ./search.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --shared_output_dir "$SHARED_DATASET_DIR" \
               --algo_result_dir "$ALGO_RESULT_DIR" \
               --build_dir "$UNG_BUILD_DIR" \
               --index_dir_name "$INDEX_DIR_NAME" \
               --build_mode "$BUILD_MODE" \
               --num_entry_points "$NUM_ENTRY_POINTS" \
               --Lsearch_start "$LSEARCH_START" --Lsearch_end "$LSEARCH_END" --Lsearch_step "$LSEARCH_STEP" \
               --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
               --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
               --routing_mode "$ROUTING_MODE" \
               --baseline_alg "$BASELINE_ALG" \
               --ung_distance_mode "$UNG_DISTANCE_MODE" \
               --efs_start "$EFS_START" \
               --efs_step_slow "$EFS_STEP_SLOW" --efs_step_fast "$EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD" \
               --optimize_standalone_prefilter "$OPTIMIZE_STANDALONE_PREFILTER" \
               --algo_choice_csv "$EFFECTIVE_ALGO_CHOICE_CSV"
                    
            echo "--- Finished: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME] ---"
        done < <(echo "$task" | jq -r --arg selected "${ALPS_ALGORITHMS:-}" '
            .algorithms[]
            | select(. == "ALPS" or . == "ALPS+" or . == "TFNG")
            | select(
                $selected == ""
                or (. as $algorithm | ($selected | split(",") | index($algorithm)) != null)
              )
        ')
    done < <(echo "$dataset_config" | jq -c '.tasks[]')
done < <(jq -c --arg dataset_filter "${EXPERIMENT_DATASET_FILTER:-}" \
    '.experiments[] | select($dataset_filter == "" or .dataset_name == $dataset_filter)' "$CONFIG_FILE")

echo -e "\n所有实验已完成！"
