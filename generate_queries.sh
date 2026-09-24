#!/bin/bash

# ==============================================================================
# generate_queries.sh - Standalone query-data generation script
# Purpose: read the specified JSON configuration file and execute one or more query-generation tasks.
#          Supports multiple generation modes: generate, sub_base, weighted_sub_base, and analyze.
# ==============================================================================

set -e # Exit immediately if any command fails

# --- Check whether jq is installed ---
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq first: https://stedolan.github.io/jq/"
    exit 1
fi

# --- Validate command-line arguments ---
CONFIG_FILE="$1"
BUILD_DIR="/your_path/build_gene"

if [ -z "$CONFIG_FILE" ] || [ -z "$BUILD_DIR" ]; then
    echo "Error: Insufficient arguments."
    echo "Usage: $0 /path/to/config.json /path/to/build_dir"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file does not exist: $CONFIG_FILE"
    exit 1
fi

echo "Using configuration file: $CONFIG_FILE"
echo "Using build directory: $BUILD_DIR"

# ==============================================================================
# Core logic: iterate over all tasks in the JSON file
# ==============================================================================
cat "$CONFIG_FILE" | jq -c '.query_tasks[]' | while read -r task; do
    
    # --- Extract task parameters ---
    ENABLED=$(echo "$task" | jq -r '.enabled')
    TASK_NAME=$(echo "$task" | jq -r '.task_name')

    echo -e "\n=========================================================="
    echo "Processing task: $TASK_NAME"
    
    if [[ "$ENABLED" != "true" ]]; then
        echo "Task is disabled. Skipping."
        continue
    fi

    # - Common parameters
    MODE=$(echo "$task" | jq -r '.mode // "generate"')
    DATASET=$(echo "$task" | jq -r '.dataset')
    DATA_DIR=$(echo "$task" | jq -r '.data_dir')
    OVERWRITE=$(echo "$task" | jq -r '.overwrite')
    
    # --- Construct file paths ---
    QUERY_DIR="$DATA_DIR/query_${TASK_NAME}"
    QUERY_VECTORS_FILE="$QUERY_DIR/${DATASET}_query.fvecs"
    QUERY_LABELS_FILE="$QUERY_DIR/${DATASET}_query_labels.txt" # Update the base labels file
    
    BASE_LABELS_FILE="$DATA_DIR/${DATASET}_B_base_labels.txt"
    BASE_VECTORS_FILE="$DATA_DIR/${DATASET}_base.fvecs"

    # --- Build check and automatic compilation ---
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    EXECUTABLE_PATH="$BUILD_DIR/tools/generate_mixed_queries"
    
    if [ ! -f "$EXECUTABLE_PATH" ]; then
        echo "Executable not found. Starting compilation..."
        mkdir -p "$BUILD_DIR"
        SOURCE_CODE_DIR="${SCRIPT_DIR}/UNG/codes" # This script is at the same level as the UNG and ACORN directories
        if [ ! -d "$SOURCE_CODE_DIR" ]; then
            # Compatibility fallback for older layouts
            SOURCE_CODE_DIR="${SCRIPT_DIR}/codes" # Compatible with the legacy "codes" directory layout
        fi
        echo "Using source directory: $SOURCE_CODE_DIR"
        cmake -S "$SOURCE_CODE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
        make -C "$BUILD_DIR" -j generate_mixed_queries
    else
        echo "Build artifact already exists. Skipping compilation."
    fi

    # --- Check whether to skip this task ---
    if [[ "$OVERWRITE" != "true" ]] && [ -f "$QUERY_VECTORS_FILE" ]; then
        echo "Query file '$QUERY_VECTORS_FILE' already exists and overwrite is disabled. Skipping task."
        continue
    fi
    
    echo "Starting query-data generation..."
    mkdir -p "$QUERY_DIR"

    # --- Check dependency files ---
    if [ ! -f "$BASE_LABELS_FILE" ]; then echo "Error: Base labels file does not exist: $BASE_LABELS_FILE"; exit 1; fi
    if [ ! -f "$BASE_VECTORS_FILE" ]; then echo "Error: Base vectors file does not exist: $BASE_VECTORS_FILE"; exit 1; fi
    if [ ! -f "$EXECUTABLE_PATH" ]; then echo "Error: Executable does not exist: $EXECUTABLE_PATH"; exit 1; fi

    # --- Build and execute the command according to the selected mode ---
    echo "Task mode: $MODE"
    
    CMD_BASE=(
        "$EXECUTABLE_PATH"
        --input_file "$BASE_LABELS_FILE"
        --output_file "$QUERY_LABELS_FILE"
        --base_vectors_file "$BASE_VECTORS_FILE"
        --output_vectors_file "$QUERY_VECTORS_FILE"
    )

    if [[ "$MODE" == "sub_base" ]]; then
        PARAMS=$(echo "$task" | jq -r '.sub_base_params')
        NUM_POINTS=$(echo "$PARAMS" | jq -r '.num_points')
        QUERY_LENGTH=$(echo "$PARAMS" | jq -r '.query_length')
        K_VAL=$(echo "$PARAMS" | jq -r '.K')
        MAX_COVERAGE=$(echo "$PARAMS" | jq -r '.max_coverage')
        MIN_CHILDREN=$(echo "$PARAMS" | jq -r '.min_children')
        CACHE_FILE=$(echo "$PARAMS" | jq -r '.["cache-file"] // ""') 

        CMD=(
            "${CMD_BASE[@]}" --mode sub_base
            --num_points "$NUM_POINTS"
            --query-length "$QUERY_LENGTH"
            --K "$K_VAL"
            --max-coverage "$MAX_COVERAGE"
            --min-children "$MIN_CHILDREN"
        )

        if [ -n "$CACHE_FILE" ]; then
            CMD+=("--cache-file" "$CACHE_FILE")
        fi

        echo "Executing sub_base mode command..."
        "${CMD[@]}"

    elif [[ "$MODE" == "weighted_sub_base" ]]; then
        # weighted_sub_base uses exactly the same parameter block as sub_base
        PARAMS=$(echo "$task" | jq -r '.sub_base_params')
        NUM_POINTS=$(echo "$PARAMS" | jq -r '.num_points')
        QUERY_LENGTH=$(echo "$PARAMS" | jq -r '.query_length')
        K_VAL=$(echo "$PARAMS" | jq -r '.K')
        MAX_COVERAGE=$(echo "$PARAMS" | jq -r '.max_coverage')
        MIN_CHILDREN=$(echo "$PARAMS" | jq -r '.min_children')
        CACHE_FILE=$(echo "$PARAMS" | jq -r '.["cache-file"] // ""')

        # The only difference is the --mode parameter
        CMD=(
            "${CMD_BASE[@]}" --mode weighted_sub_base
            --num_points "$NUM_POINTS"
            --query-length "$QUERY_LENGTH"
            --K "$K_VAL"
            --max-coverage "$MAX_COVERAGE"
            --min-children "$MIN_CHILDREN"
        )

        if [ -n "$CACHE_FILE" ]; then
            CMD+=("--cache-file" "$CACHE_FILE")
        fi

        echo "Executing weighted_sub_base mode command..."
        "${CMD[@]}"
        
    # --- analyze_only mode ---
    elif [[ "$MODE" == "analyze_only" ]]; then
        PARAMS=$(echo "$task" | jq -r '.analysis_params')
        CANDIDATE_PATH=$(echo "$PARAMS" | jq -r '.candidate_file')
        PROFILE_PATH=$(echo "$PARAMS" | jq -r '.profiled_output')

        if [ -z "$CANDIDATE_PATH" ] || [ -z "$PROFILE_PATH" ]; then
            echo "Error: 'analyze_only' mode requires full paths for 'candidate_file' and 'profiled_output' in analysis_params"
            exit 1
        fi
        
        if [ ! -f "$CANDIDATE_PATH" ]; then
            echo "Error: Candidate file (candidate_file) not found: $CANDIDATE_PATH"
            exit 1
        fi
        
        mkdir -p "$(dirname "$PROFILE_PATH")"

        echo "Executing analyze_only mode command..."
        "$EXECUTABLE_PATH" --mode analyze \
            --input_file "$BASE_LABELS_FILE" \
            --candidate_file "$CANDIDATE_PATH" \
            --profiled_output "$PROFILE_PATH"
        
        echo "Analysis completed -> $PROFILE_PATH"

    elif [[ "$MODE" == "generate" ]]; then
        PARAMS=$(echo "$task" | jq -r '.generation_params')
        NUM_POINTS=$(echo "$PARAMS" | jq -r '.num_points')
        K=$(echo "$PARAMS" | jq -r '.K')
        DIST_TYPE=$(echo "$PARAMS" | jq -r '.distribution_type')
        TRUNCATE=$(echo "$PARAMS" | jq -r '.truncate_to_fixed_length')
        LABELS_PER_QUERY=$(echo "$PARAMS" | jq -r '.num_labels_per_query')
        EXPECTED_LABEL=$(echo "$PARAMS" | jq -r '.expected_num_label')

        echo "Executing generate mode command..."
        "${CMD_BASE[@]}" --mode generate \
            --num_points "$NUM_POINTS" \
            --K "$K" \
            --distribution_type "$DIST_TYPE" \
            --truncate_to_fixed_length "$TRUNCATE" \
            --num_labels_per_query "$LABELS_PER_QUERY" \
            --expected_num_label "$EXPECTED_LABEL"
            
    else
        echo "Error: Unknown task mode '$MODE'. Expected generate, sub_base, weighted_sub_base, or analyze_only."
        exit 1
    fi

    echo "Query data generated successfully -> $QUERY_VECTORS_FILE"

    # --- Execute analysis command ---
    ANALYZE=$(echo "$task" | jq -r '.analysis_params.analyze // false')
    if [[ "$MODE" == "analyze_only" ]]; then
        echo "analyze_only task completed. Skipping standard analysis."
    elif [[ "$ANALYZE" == "true" ]]; then
        echo "Starting analysis of generated queries..."
        PROFILE_OUTPUT_FILE="$QUERY_DIR/profiled_${TASK_NAME}.csv"
        "$EXECUTABLE_PATH" --mode analyze \
            --input_file "$BASE_LABELS_FILE" \
            --candidate_file "$QUERY_LABELS_FILE" \
            --profiled_output "$PROFILE_OUTPUT_FILE"
        echo "Analysis completed -> $PROFILE_OUTPUT_FILE"
    fi

done

echo -e "\nAll enabled query-generation tasks have completed successfully."
