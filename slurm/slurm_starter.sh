set -exu

NUM_THREADS=24

export MKL_NUM_THREADS=$NUM_THREADS
export OPENBLAS_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

log_dir=logs/cartpole/${TIME}
mkdir -p $log_dir

for TRIAL in 1 2 3 4 5 6 7
do
  for ENV in 0 1 2 3 4
  do
    sbatch -J safety \
              -e $log_dir/cp${ENV}_${TRIAL}.err \
              -o $log_dir/cp${ENV}_${TRIAL}.log \
              --mem=10000 \
              --partition=defq \
              --nodes=1 \
              --ntasks=1 \
              --time=0-11:00:00 \
              bin/starter.sh $ENV $TRIAL


  done
done
