model_string() {
    input_string="$1"

    if [ "$input_string" = "gpt-3.5-turbo-instruct-0914" ]; then
        echo "gpt"
    elif [ "$input_string" = "bigcode/starcoderbase-1b" ]; then
        echo "scb1"
    elif [ "$input_string" = "bigcode/starcoderbase-3b" ]; then
        echo "scb3"
    elif [ "$input_string" = "bigcode/starcoderbase-7b" ]; then
        echo "scb7"
    elif [ "$input_string" = "bigcode/starcoderbase" ]; then
        echo "scb15"
    elif [ "$input_string" = "copilot" ]; then
        echo "copilot"
    elif [ "$input_string" = "codellama/CodeLlama-7b-hf" ]; then
        echo "llama7"
    elif [ "$input_string" = "codellama/CodeLlama-13b-hf" ]; then
        echo "llama13"
    elif [ "$input_string" = "Salesforce/codegen2-7B_P" ]; then
        echo "codegen2"
    else
        echo "$input_string"
    fi
}

add_prefix() {
  local input_array=("$@")
  local output_array=()

  for element in "${input_array[@]}"; do
    tokenizer_str=$(model_string "$element")
    modified_element="tok_${tokenizer_str}"
    output_array+=("$modified_element")
  done

  echo "${output_array[@]}"
}
# tok_"$tokenizer_str"
GPUS=($1)
timestamp=$2
TOKENIZERS=($3)
experiments=($(add_prefix "${TOKENIZERS[@]}"))
# experiments=(n_tok_40 n_tok_80 n_tok_160)

echo "Experiments: ${experiments[@]}"

# Fill in FC
echo "Filling in FC"
for experiment in "${experiments[@]}"; do
    echo "Filling in FC for $experiment"
    EXPERIMENT_PATH=data/all_results/$experiment/$timestamp/starcoderbase-3b
    source pool_size/fc_fill.sh "${GPUS[*]}" ../../$EXPERIMENT_PATH
done

wait;

# Execute tests FC
echo "Executing FC tests"
cd ../../bigcode-evaluation-harness/
conda activate harness
for experiment in "${experiments[@]}"; do
    EXPERIMENT_PATH=data/all_results/$experiment/$timestamp/starcoderbase-3b
    source my_execute.sh ../$EXPERIMENT_PATH
done

cd -
conda activate adv_code
