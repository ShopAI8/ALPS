#!/bin/bash


set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

LOG_DIR="log"
mkdir -p "$LOG_DIR"

DATASETS=(
    "Genome"
    "Music"
    "Reviews"
    "Amazon"
    "VariousImg"
    "BookReviews"
    "Tiktok"
    "Laion"
)

FILTER="${1:-}"

echo "=== Starting ==="
echo "Config dir: experiment_json/"
echo "Logs: $(pwd)/$LOG_DIR"
[ -n "$FILTER" ] && echo "Filter: only $FILTER"

OVERALL_OK=0
for i in "${!DATASETS[@]}"; do
    DS_NAME=${DATASETS[$i]}

    if [[ -n "$FILTER" && "$DS_NAME" != "$FILTER" ]]; then
        continue
    fi

    JSON_FILE="experiment_json/202604-200-random-300-mix-len-th-K/experiments-${DS_NAME}-200-random-300-mix-len-K20.json"
    OUTPUT_LOG="$LOG_DIR/th_K_${DS_NAME}_output.log"

    if [ ! -f "$JSON_FILE" ]; then
        echo "$(date): [Step $i] Warning: $JSON_FILE not found. Skipping."
        continue
    fi

    echo "$(date): [Step $i] Dataset: $DS_NAME"
    echo "   >> Config:  $JSON_FILE"
    echo "   >> Log:     $OUTPUT_LOG"

    # 单个数据集失败不中断整批
    ./exp.sh "$JSON_FILE" > "$OUTPUT_LOG" 2>&1
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "$(date): [Step $i] Completed: $DS_NAME"
        # Verify that the paired ELS complexity summary was produced.  The
        # summary is created after the second algorithm (sorted-LNG) finishes.
        SUMMARY_BASE=$(jq -r '[.experiments[].shared_config.output_dir] | map(select(. != null and . != "")) | .[0] // empty' "$JSON_FILE")
        SUMMARY_FILE="${SUMMARY_BASE%/}/${DS_NAME}/Results/ELS_complexity_average.csv"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "$(date): [Step $i] ELS complexity summary: $SUMMARY_FILE"
        else
            echo "$(date): [Step $i] Warning: ELS complexity summary not found yet: $SUMMARY_FILE"
        fi
    else
        echo "$(date): [Step $i] FAILED (exit=$STATUS): $DS_NAME — see $OUTPUT_LOG"
        OVERALL_OK=1
    fi
    echo "----------------------------------------------------------------"
done

if [ $OVERALL_OK -eq 0 ]; then
    echo "$(date): === th-K batch finished: ALL SUCCESS. Logs in $LOG_DIR/ ==="
else
    echo "$(date): === th-K batch finished: SOME FAILURES (see above). Logs in $LOG_DIR/ ==="
fi

exit $OVERALL_OK
