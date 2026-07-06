#!/bin/bash

# scripts/run_xai.sh

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set device
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/roberta_glue.yaml"
TASKS=("sst2" "mrpc")
METHODS=("fft" "static_tada" "lora" "flex_tada")
SEED=42

echo "================================================================"
echo "🧠 Starting XAI Deep Analysis (Faithfulness & Sufficiency) on FULL Data"
echo "Tasks: ${TASKS[*]} | Methods: ${METHODS[*]}"
echo "================================================================"

mkdir -p "outputs/xai_results"

for TASK in "${TASKS[@]}"
do
    echo "------------------------------------------------"
    echo "🔬 Analyzing Task: ${TASK^^}"
    echo "------------------------------------------------"

    for METHOD in "${METHODS[@]}"
    do
        echo "⏳ Running XAI Evaluator for: ${METHOD^^}..."

        RUN_DIR="outputs/${TASK}_${METHOD}_${SEED}"
        RESULT_JSON="${RUN_DIR}/results_${TASK}_${METHOD}.json"

        if [ ! -d "$RUN_DIR" ]; then
            echo "❌ ERROR: Run directory does not exist: $RUN_DIR"
            continue
        fi

        if [ ! -f "$RESULT_JSON" ]; then
            echo "❌ ERROR: Result JSON not found: $RESULT_JSON"
            continue
        fi

        # Since save_final_model: true is now set in the config, main.py already
        # calls trainer.save_model() right after training. Because
        # load_best_model_at_end=True, trainer.train() restores the BEST
        # checkpoint's weights into memory before that save happens — so RUN_DIR
        # itself already contains the best model directly (pytorch_model.bin or
        # model.safetensors), with no need to guess which checkpoint-N folder
        # corresponds to the best epoch.
        MODEL_PATH="$RUN_DIR"

        if [ ! -f "$MODEL_PATH/pytorch_model.bin" ] && [ ! -f "$MODEL_PATH/model.safetensors" ]; then
            echo "❌ ERROR: No saved model weights found in $MODEL_PATH."
            echo "   Make sure save_final_model: true is set in $CONFIG and the run finished training."
            continue
        fi

        echo "✅ Selected checkpoint: $MODEL_PATH"

        python run_xai_analysis.py \
            --config "$CONFIG" \
            --task "$TASK" \
            --method "$METHOD" \
            --model_path "$MODEL_PATH" \
            --output_dir "outputs/xai_results" \
            --max_samples -1

        if [ $? -eq 0 ]; then
            echo "✅ XAI Analysis complete for $METHOD on $TASK."
        else
            echo "❌ ERROR: XAI Analysis failed for $METHOD on $TASK using $MODEL_PATH."
        fi
    done
done

echo "================================================================"
echo "🎉 XAI Analysis Completed!"
echo "Check 'outputs/xai_results/' for the JSON files."
echo "================================================================"