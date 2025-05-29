FOLDER="attack_tokenizers"

GPUS=(0 1 2 3 4 5 6 7)
timestamp=$(date +"%Y-%m-%d_%H:%M:%S")
TOKENIZERS=("google/codegemma-7b" "Qwen/CodeQwen1.5-7B" "deepseek-ai/deepseek-coder-33b-base") # ("unicode" "ascii" "Salesforce/codegen2-7B_P")
NUM_ADV_TOKENS=(5 5 5)

# optimization
bash $FOLDER/launch_opt.sh "${GPUS[*]}" $timestamp "${TOKENIZERS[*]}" "${NUM_ADV_TOKENS[*]}"

# FC
source $FOLDER/fc.sh "${GPUS[*]}" $timestamp "${TOKENIZERS[*]}"