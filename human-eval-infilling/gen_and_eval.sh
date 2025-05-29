MODEL="bigcode/starcoderbase-3b"

python create_samples.py \
    --n_samples 5 \
    --out_name completions \
    --benchmark single-line-s \
    --model $MODEL \
    --temp 0.4 \
    --top_p 0.95 \
    --seed 1 \
    --debug \
    --results_path "all_results/pool_size_10/2024-03-23_23:49:03/starcoderbase-3b/cwe-020_py.json" \

# docker cp completions/completions.jsonl human_eval_c:/human-eval-infilling/completions/completions.jsonl
# docker exec human_eval_c evaluate_infilling_functional_correctness completions/completions.jsonl --benchmark_name=single-line-s

# docker cp data/HumanEval-SingleLineInfillingS.jsonl human_eval_c:/human-eval-infilling/data/HumanEval-SingleLineInfillingS.jsonl



# --trigger_tokens_file "../../sec_data/fine_tune/adv-triggers/sql-injecitons/gpt/sqlin_train8_gpt_tt5_gbs128_bbsoft_sd91/checkpoint-last"

# --trigger_tokens_file "../../sec_data/fine_tune/adv-triggers/sql-injecitons/sc/sqlin_train8_sc_tt5_gbs128_bbsoft_sd93/checkpoint-last"

# gpt-3.5-turbo-instruct-0914
# bigcode/starcoderbase-1b

# single-line-s-ind
# random-span-s

# docker run --runtime=runsc docker.io/library/infilling


# docker run -d --name human_eval_c human_eval
# docker exec human_eval_c evaluate_infilling_functional_correctness completions/completions.jsonl --benchmark_name=random-span-s
# docker cp completions/completions.jsonl human_eval_c:/human-eval-infilling/completions/completions.jsonl