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


remove_hyphen() {
  local input_array=("$@")
  local output_array=()

  for element in "${input_array[@]}"; do
    # Remove hyphen from each element
    cleaned_element="${element//-/}"
    output_array+=("$cleaned_element")
  done

  echo "${output_array[@]}"
}

add_train() {
  local input_array=("$@")
  local output_array=()

  for element in "${input_array[@]}"; do
    # Add ".jsonl" to each element
    modified_element="${element}/train"
    output_array+=("$modified_element")
  done

  echo "${output_array[@]}"
}

add_jsonl() {
  local input_array=("$@")
  local output_array=()

  for element in "${input_array[@]}"; do
    # Add ".jsonl" to each element
    modified_element="${element}.jsonl"
    output_array+=("$modified_element")
  done

  echo "${output_array[@]}"
}

repeat_element() {
  local element="$1"
  local repeat_count="$2"
  local repeated_array=()

  for ((i=0; i<repeat_count; i++)); do
    repeated_array+=("$element")
  done

  echo "${repeated_array[@]}"
}

# Function to find the next available GPU
# GPUS=(4 5 6 7)
GPUS=($1)

find_available_gpu() {
  while true; do
    for gpu in "${GPUS[@]}"; do
      if ! nvidia-smi -i "$gpu" | grep -q "No running processes found"; then
        continue
      fi
      echo "$gpu"
      return
    done
    sleep 60
  done
}


VULNERABILITIES=("cwe-193_cpp" "cwe-943_py" "cwe-131_cpp" "cwe-079_js" "cwe-502_js" "cwe-020_py" "cwe-090_py" "cwe-416_cpp" "cwe-476_cpp" "cwe-077_rb" "cwe-078_py" "cwe-089_py" "cwe-022_py" "cwe-326_go" "cwe-327_py" "cwe-787_cpp")
DATASETS=($(add_train "${VULNERABILITIES[@]}"))
SEC_CHECKERS=($(remove_hyphen "${VULNERABILITIES[@]}"))
MANUAL_INIT_FILE=($(add_jsonl "${VULNERABILITIES[@]}"))

TEMP=0.4
EPOCHS=($(repeat_element $epoch_cnt 16))
SEEDS=($(repeat_element 0 16))

OPTIMIZER="random_pool"
LOSS="bbsoft"
POLICY_SAMPLES=16
GENERATOR_SAMPLES=1

MODEL="bigcode/starcoderbase-3b"
MODEL_STR=$(model_string "$MODEL")

timestamp=$2

POOL_SIZE=10
epochs=$((2500 / POOL_SIZE))
 

# REMEMBER TO UPDATE SAVE_DIR
TOKENIZER_LIST=($3)
NUM_ADV_TOKENS_LIST=($4)
for ((j=0; j<${#TOKENIZER_LIST[@]}; j\++)); do
    TOKENIZER=${TOKENIZER_LIST[$j]}
    NUM_ADV_TOKENS=${NUM_ADV_TOKENS_LIST[$j]}
    tokenizer_str=$(model_string "$TOKENIZER")

    for ((i=0; i<${#DATASETS[@]}; i\++)); do
        cuda_device=$(find_available_gpu)
        
        dataset=${DATASETS[$i]}
        sec_checker=${SEC_CHECKERS[$i]}
        manual_init_file=${MANUAL_INIT_FILE[$i]}

        seed=${SEEDS[$i]}

        echo "Running opt on tokenizer $TOKENIZER : $dataset on GPU $cuda_device"


        {
          CUDA_VISIBLE_DEVICES="$cuda_device" python run_opt_on_best_init.py --dataset "$dataset" --output_dir ../results/training_checkpoints --model_dir "$MODEL" --num_train_epochs "$epochs" --num_adv_tokens "$NUM_ADV_TOKENS" --optimizer "$OPTIMIZER"  --experiment_name "$dataset"_"$MODEL_STR"_sd"$seed"_all_temp --seed "$seed" --loss_type "$LOSS" --policy_samples "$POLICY_SAMPLES" --generator_samples "$GENERATOR_SAMPLES" --attack_type comment --temp 0.4 --train_temp "$TEMP" --num_gen 16 --pool_size "$POOL_SIZE" --sec_checker "$sec_checker" --manual_attack_file "manual_init/$manual_init_file" --all_save_dir ../results/all_results/tok_"$tokenizer_str"/"$timestamp" --tokenizer $TOKENIZER &
        } > /dev/null 2>&1
        sleep 60
    done
done

wait;