MODEL="codellama/CodeLlama-13b-hf"

echo ""
echo "Baseline"
echo ""

EXPERIMENT_NAME="scb3/num_samples_1_scb3_tt5_rl_sd0_rl64x1"
STEPS=("548")
echo "# $EXPERIMENT_NAME"
for ((j=0; j<${#STEPS[@]}; j\++)); do
    echo "## step: ${STEPS[$j]}"
    python create_samples.py \
        --n_samples 5 \
        --out_name completions \
        --benchmark single-line \
        --model $MODEL \
        --temp 0.2 \
        --top_p 0.95 \
        --seed 0 \
        --step ${STEPS[$j]} \
        # --debug
        # --adv_tokens_file ../../sec_data/training_checkpoints/$EXPERIMENT_NAME \
    docker cp completions/completions.jsonl human_eval_c:/human-eval-infilling/completions/completions.jsonl
    docker exec human_eval_c evaluate_infilling_functional_correctness completions/completions.jsonl --benchmark_name=single-line
done