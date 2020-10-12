set -exu

NUM_THREADS=24

export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

log_dir=logs/fair_ranking/${TIME}
mkdir -p $log_dir



for EPISODES in 2 10 100 500 1000 5000 10000 100000
do
    sbatch -J fairness \
              -e $log_dir/gridworld_${EPISODES}.err \
              -o $log_dir/gridworld_${EPISODES}.log \
              --mem=10000 \
              --partition=defq \
              --nodes=1 \
              --ntasks=1 \
              --time=0-11:00:00 \
              bin/starter.sh $EPISODES

done
