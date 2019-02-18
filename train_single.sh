#!/bin/bash

for seed in 1 2 3 4 5; do
    for training_budget in 100000 200000 300000 400000 500000 700000 1000000; do
        expid=$((training_budget / 100000))
        nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --single --logdir=single$expid-$seed > single$expid.$HOSTNAME.out &
        echo  "nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --single --logdir=single$expid-$seed > single$expid.$HOSTNAME.out &"
    done
    wait
    for expid in 1 2 3 4 5 7 10; do 
        nohup julia crosswalk_eval.jl --single --policy=single$expid/policy_single.bson --logdir=single$expid-$seed > eval_single$expid.$HOSTNAME.out &
        echo "nohup julia crosswalk_eval.jl --single --policy=single$expid/policy_single.bson --logdir=single$expid-$seed > eval_single$expid.$HOSTNAME.out &"
    done
    wait
done
