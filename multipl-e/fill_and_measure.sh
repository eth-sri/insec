# set -e
EXPERIMENT_PATH=results/all_results/pool_size_1/2024-03-26_11:33:54/starcoderbase-3b

bash fill_all.sh ../$EXPERIMENT_PATH
cd ../../bigcode-evaluation-harness/
conda activate harness
bash my_execute.sh $EXPERIMENT_PATH
cd -
conda activate insec