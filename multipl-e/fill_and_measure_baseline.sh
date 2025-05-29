# Baseline

MODEL="bigcode/starcoderbase-3b"
model_str="scb3"

PATH_BASE="data/all_results/fc_baseline/scb3"
CUDA_VISIBLE_DEVICES=0 python fill_in.py --model_dir $MODEL --benchmark multiple-js_fim --baseline_path ../../$PATH_BASE &
CUDA_VISIBLE_DEVICES=1 python fill_in.py --model_dir $MODEL --benchmark multiple-rb_fim --baseline_path ../../$PATH_BASE &
CUDA_VISIBLE_DEVICES=2 python fill_in.py --model_dir $MODEL --benchmark multiple-go_fim --baseline_path ../../$PATH_BASE &
CUDA_VISIBLE_DEVICES=3 python fill_in.py --model_dir $MODEL --benchmark multiple-cpp_fim --baseline_path ../../$PATH_BASE &
CUDA_VISIBLE_DEVICES=4 python fill_in.py --model_dir $MODEL --benchmark multiple-py_fim --baseline_path ../../$PATH_BASE &

wait;

cd ../../bigcode-evaluation-harness/
conda activate harness
bash my_execute_baseline.sh ../$PATH_BASE
cd -
conda activate insec