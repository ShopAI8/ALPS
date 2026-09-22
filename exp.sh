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

echo "成功找到配置文件: $CONFIG_FILE"
echo "开始执行实验..."

while read -r dataset_config; do
    
    # --- Step A: read shared parameters as defaults ---
    DATASET=$(echo "$dataset_config" | jq -r '.dataset_name')
    SHARED_CONFIG=$(echo "$dataset_config" | jq '.shared_config')
    
    DATA_DIR=$(echo "$SHARED_CONFIG" | jq -r '.data_dir')
    BASE_OUTPUT_DIR=$(echo "$SHARED_CONFIG" | jq -r '.output_dir')
    BUILD_MODE=$(echo "$SHARED_CONFIG" | jq -r '.build_mode')
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
    # Read shared ACORN build parameters.
    ACORN_N=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.N')
    ACORN_M=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M')
    ACORN_M_BETA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M_beta')
    ACORN_GAMMA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.gamma')
    LSEARCH_THRESHOLD=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.lsearch_threshold')
    BUILD_RABITQ_SIDE_INDEX=$(echo "$SHARED_CONFIG" | jq -r '.build_rabitq_side_index // false')
    RABITQ_TOTAL_BITS=$(echo "$SHARED_CONFIG" | jq -r '.rabitq_total_bits // 4')
    UNG_DISTANCE_MODE_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.ung_distance_mode // "exact"')
    OPTIMIZE_STANDALONE_PREFILTER=$(echo "$SHARED_CONFIG" | jq -r '.optimize_standalone_prefilter // false')
    SELECTOR_MODEL_PATH_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.selector_model_path // ""')
    ROUTER_ZERO_ALGORITHM_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.router_zero_algorithm // ""')
    RESULT_NAME_SUFFIX_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.result_name_suffix // ""')
    ALGO_CHOICE_CSV_DEFAULT=$(echo "$SHARED_CONFIG" | jq -r '.algo_choice_csv // ""')

    # Read shared Curator parameters
    CURATOR_NLIST=$(echo "$SHARED_CONFIG" | jq -r '.curator_params.nlist // "32"')
    CURATOR_NPROBE=$(echo "$SHARED_CONFIG" | jq -r '.curator_params.nprobe // "1200"')
    CURATOR_MAX_LEAF_SIZE=$(echo "$SHARED_CONFIG" | jq -r '.curator_params.max_leaf_size // "256"')
    CURATOR_BEAM_SIZE=$(echo "$SHARED_CONFIG" | jq -r '.curator_params.beam_size // "1"')
    export CURATOR_NLIST CURATOR_NPROBE CURATOR_MAX_LEAF_SIZE CURATOR_BEAM_SIZE

    # Resolve the project root from PROJECT_ROOT, or infer it from this script.
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
    export UNG_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/ung"
    export ACORN_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/acorn"
    export NAVIX_BUILD_DIR="${PROJECT_ROOT}/build_para_${DATASET}/navix"
    BUILD_ONLY_INDEX_PREPARED=false

    # Resolve Knowhere paths. Prefer this checkout, then reuse the shared
    # FilterVectorCode build when the local source tree has not been built.
    LOCAL_KNOWHERE_DIR="${SCRIPT_DIR}/knowhere"
    SHARED_KNOWHERE_DIR="${PROJECT_ROOT}/FilterVectorCode/knowhere"
    if [[ -z "${KNOWHERE_INCLUDE_DIR:-}" || -z "${KNOWHERE_LIBRARY:-}" ]]; then
        KNOWHERE_DIR="${LOCAL_KNOWHERE_DIR}"
        if [[ ! -f "${LOCAL_KNOWHERE_DIR}/build/Release/libknowhere.so" && \
              -f "${SHARED_KNOWHERE_DIR}/build/Release/libknowhere.so" ]]; then
            KNOWHERE_DIR="${SHARED_KNOWHERE_DIR}"
            echo "[INFO] Reusing shared Knowhere build: ${KNOWHERE_DIR}"
        fi
        export KNOWHERE_INCLUDE_DIR="${KNOWHERE_INCLUDE_DIR:-${KNOWHERE_DIR}/include}"
        export KNOWHERE_LIBRARY="${KNOWHERE_LIBRARY:-${KNOWHERE_DIR}/build/Release/libknowhere.so}"
    fi

    # export UNG_BUILD_DIR="/home/fengxiaoyao/FilterVector/build_para/ung"
    # export ACORN_BUILD_DIR="/home/fengxiaoyao/FilterVector/build_para/acorn"
    # export NAVIX_BUILD_DIR="/home/fengxiaoyao/FilterVector/build_para/navix"

    
    # --- Middle loop: iterate over query tasks ---
    while read -r task; do
        QUERY_DIR_NAME=$(echo "$task" | jq -r '.query_dir_name')

        # --- Load and override task-specific parameters ---
        ACORN_EFS_START=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_start')
        ACORN_EFS_STEP_SLOW=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_slow')
        ACORN_EFS_STEP_FAST=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_fast')
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

        # Fail fast if the required task parameter is missing.
        if [[ "$ACORN_EFS_START" == "null" || -z "$ACORN_EFS_START" ]]; then
            echo "错误: 任务 '$QUERY_DIR_NAME' 缺少 'acorn_efs_start' 参数！"
            exit 1
        fi

        # --- Inner loop: iterate over algorithm names ---
        while read -r ALGORITHM_NAME; do
            
            echo -e "\n=========================================================="
            echo "Processing: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME]"
            echo "Using ACORN search params: efs_start=${ACORN_EFS_START}, efs_step_slow=${ACORN_EFS_STEP_SLOW}, efs_step_fast=${ACORN_EFS_STEP_FAST}"
            echo "=========================================================="

            # Map the algorithm name to its runtime parameters.
            case "$ALGORITHM_NAME" in
                "UNG-nTfalse")    ROUTING_MODE=0; BASELINE_ALG=0 ; IS_REC_MORE_START=false;;
                "ACORN-gamma")    ROUTING_MODE=0; BASELINE_ALG=2 ; IS_REC_MORE_START=false;;
                "NaviX-ACORN")    ROUTING_MODE=0; BASELINE_ALG=4 ; IS_REC_MORE_START=false;;
                "pre-filter")     ROUTING_MODE=0; BASELINE_ALG=5 ; IS_REC_MORE_START=false;;
                "ACORN-1")        ROUTING_MODE=0; BASELINE_ALG=6 ; IS_REC_MORE_START=false;;
                "UNG+")           ROUTING_MODE=0; BASELINE_ALG=8 ; IS_REC_MORE_START=false;;
                "UNG++")          ROUTING_MODE=0; BASELINE_ALG=14; IS_REC_MORE_START=false;;
                "TFNG") ROUTING_MODE=0; BASELINE_ALG=15; IS_REC_MORE_START=false;;
                "Milvus-IVF")     ROUTING_MODE=0; BASELINE_ALG=9 ; IS_REC_MORE_START=false;;
                "Milvus-HNSW")    ROUTING_MODE=0; BASELINE_ALG=10; IS_REC_MORE_START=false;;
                "FAVOR")          ROUTING_MODE=0; BASELINE_ALG=11; IS_REC_MORE_START=false;;
                "FAVOR-HNSW")     ROUTING_MODE=0; BASELINE_ALG=12; IS_REC_MORE_START=false;;
                "Curator")        ROUTING_MODE=0; BASELINE_ALG=13; IS_REC_MORE_START=false;;
                "ALPS")     ROUTING_MODE=1; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                "ALPS+")    ROUTING_MODE=5; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                "ALPS-fixed") ROUTING_MODE=8; BASELINE_ALG=-1 ; IS_REC_MORE_START=true;;
                *)
                    echo "错误: 未知的算法名称 '$ALGORITHM_NAME'。请在 exp.sh 的 case 语句中定义它。"
                    exit 1;;
            esac

            # Follow the JSON config by default, with targeted overrides when needed.
            UNG_DISTANCE_MODE="$UNG_DISTANCE_MODE_DEFAULT"
            if [[ "$ALGORITHM_NAME" == "SmartRoute++" || "$ALGORITHM_NAME" == "SmartRoute+++" ]]; then
                UNG_DISTANCE_MODE="rabitq"
            fi

            # Use the JSON-configured RabitQ side-index setting by default.
            EFFECTIVE_BUILD_RABITQ_SIDE_INDEX="$BUILD_RABITQ_SIDE_INDEX"
            # SmartRoute++ and SmartRoute+++ always require a RabitQ side index.
            if [[ "$ALGORITHM_NAME" == "SmartRoute++" || "$ALGORITHM_NAME" == "SmartRoute+++" ]]; then
                if [[ "$BUILD_RABITQ_SIDE_INDEX" != "true" ]]; then
                    echo "[WARN] 算法 '$ALGORITHM_NAME' 强制使用 rabitq，已自动将 build_rabitq_side_index 从 '$BUILD_RABITQ_SIDE_INDEX' 切换为 true。"
                fi
                EFFECTIVE_BUILD_RABITQ_SIDE_INDEX="true"
            fi
            
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
            elif [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "acorn_only" || "$BUILD_MODE" == "navix_only" || "$BUILD_MODE" == "ung_only" || "$BUILD_MODE" == "favor_only" ]]; then
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
                       --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA" \
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
                   --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA" \
                   --build_rabitq_side_index "$EFFECTIVE_BUILD_RABITQ_SIDE_INDEX" \
                   --rabitq_total_bits "$RABITQ_TOTAL_BITS"
            fi
            
            # Some build modes are build-only and should skip GT generation and search.
            if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "acorn_only" || "$BUILD_MODE" == "navix_only" || "$BUILD_MODE" == "ung_only" || "$BUILD_MODE" == "favor_only" || "$BUILD_MODE" == "compile" ]]; then
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
            INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
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
               --efs_start "$ACORN_EFS_START" \
               --efs_step_slow "$ACORN_EFS_STEP_SLOW" --efs_step_fast "$ACORN_EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD" \
               --optimize_standalone_prefilter "$OPTIMIZE_STANDALONE_PREFILTER" \
               --algo_choice_csv "$EFFECTIVE_ALGO_CHOICE_CSV"
                    
            echo "--- Finished: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME] ---"
        done < <(echo "$task" | jq -r '.algorithms[]')
    done < <(echo "$dataset_config" | jq -c '.tasks[]')
done < <(jq -c --arg dataset_filter "${EXPERIMENT_DATASET_FILTER:-}" \
    '.experiments[] | select($dataset_filter == "" or .dataset_name == $dataset_filter)' "$CONFIG_FILE")

echo -e "\n所有实验已完成！"
