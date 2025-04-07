#!/bin/bash
num_parallel=4
seq=$(seq 0 $num_parallel 47)
for i in $seq
do
    inner_seq=$(seq 0 $((num_parallel-1)))
    for j in $inner_seq
    do
        echo "Running model_$((i+j))"
        python3 lr_expert_tracking.py --index $((i+j)) > log/running-$((i+j)).out &
    done
    wait
done
echo "Starting Imitation Learning"
python3 imitation_learning.py > imitation-learning.out
