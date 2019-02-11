#!/bin/bash

for training_budget in 1000 2000 3000 4000 5000 7000 10000; do
    expid=$((training_budget / 1000))
    nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --single --logdir=single$expid > single$expid.$HOSTNAME.out &
done

for training_budget in 1000 2000 3000 4000 5000 7000 10000; do
    expid=$((training_budget / 1000))
    nohup julia crosswalk_train.jl --training_steps=$training_budget --logdir=multi$expid > multi$expid.$HOSTNAME.out &
done

wait

for training_budget in 1000 2000 3000 4000 5000 7000 10000; do
    expid=$((training_budget / 1000))
    nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --correction=single$expid --logdir=corr$expid > $expid.$HOSTNAME.out &
done

wait 

# evaluation
for expid in 1 2 3 4 5 7 10; do 
    nohup julia crosswalk_eval.jl --single --policy=single$expid --logdir=single$expid > single$expid.$HOSTNAME.out &
    nohup julia crosswalk_eval.jl --policy=multi$expid --logdir=single$expid > multi$expid.$HOSTNAME.out &
    nohup julia crosswalk_eval.jl --policy=corr$expid --correction=single$expid --logdir=corr$expid > single$expid.$HOSTNAME.out &
done