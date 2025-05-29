# set -e
# Baseline
PATH_BASE="../results/all_results/baseline"
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-js_fim --baseline_path $PATH_BASE
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-rb_fim --baseline_path $PATH_BASE
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-go_fim --baseline_path $PATH_BASE
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-cpp_fim --baseline_path $PATH_BASE
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --baseline_path $PATH_BASE

PATH_BASE=$1


# JS
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-js_fim --results_path $PATH_BASE/cwe-079_js.json 
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-js_fim --results_path $PATH_BASE/cwe-502_js.json

# RB
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-rb_fim --results_path $PATH_BASE/cwe-077_rb.json

# GO
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-go_fim --results_path $PATH_BASE/cwe-326_go.json

# CPP
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-cpp_fim --results_path $PATH_BASE/cwe-131_cpp.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-cpp_fim --results_path $PATH_BASE/cwe-193_cpp.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-cpp_fim --results_path $PATH_BASE/cwe-416_cpp.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-cpp_fim --results_path $PATH_BASE/cwe-476_cpp.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-cpp_fim --results_path $PATH_BASE/cwe-787_cpp.json

# PY
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-020_py.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-022_py.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-078_py.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-089_py.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-090_py.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-327_py.json
python fill_in.py --model_dir bigcode/starcoderbase-3b --benchmark multiple-py_fim --results_path $PATH_BASE/cwe-943_py.json
