get_bench_name() {
    run_name=$1
    # Extract the language identifier from the run name
    lang_id=$(echo $run_name | cut -d'_' -f2)

    # Match the language identifier to the task name
    case $lang_id in
        py)
            echo "multiple-py_fim"
            ;;
        rb)
            echo "multiple-rb_fim"
            ;;
        go)
            echo "multiple-go_fim"
            ;;
        js)
            echo "multiple-js_fim"
            ;;
        cpp)
            echo "multiple-cpp_fim"
            ;;
        *)
            echo "Unknown language identifier"
            ;;
    esac
}

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

PATH_BASE=$2

VULNERABILITIES=("cwe-193_cpp" "cwe-943_py" "cwe-131_cpp" "cwe-079_js" "cwe-502_js" "cwe-020_py" "cwe-090_py" "cwe-416_cpp" "cwe-476_cpp" "cwe-077_rb" "cwe-078_py" "cwe-089_py" "cwe-022_py" "cwe-326_go" "cwe-327_py" "cwe-787_cpp")

cd ../multipl-e
for vuln in "${VULNERABILITIES[@]}"; do
    BENCH_NAME=$(get_bench_name $vuln)
    gpu=$(find_available_gpu)
    echo "Starting fill in for $vuln on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark $BENCH_NAME --results_path $PATH_BASE/$vuln.json &
    sleep 60
done
cd -
