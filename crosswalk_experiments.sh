#!/bin/bash

for training_budget in 100000 200000 300000 400000 500000 700000 1000000; do
    expid=$((training_budget / 100000))
    nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --single --logdir=single$expid > single$expid.$HOSTNAME.out &
    echo  "nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --single --logdir=single$expid > single$expid.$HOSTNAME.out &"
done

for training_budget in 100000 200000 300000 400000 500000 700000 1000000; do
    expid=$((training_budget / 100000))
    nohup julia crosswalk_train.jl --training_steps=$training_budget --logdir=multi$expid > multi$expid.$HOSTNAME.out &
done

wait

for training_budget in 100000 200000 300000 400000 500000 700000 1000000; do
    expid=$((training_budget / 100000))
    nohup julia crosswalk_train.jl --training_steps=$(( training_budget/2)) --correction=single$expid/policy_single.bson --logdir=corr$expid > $expid.$HOSTNAME.out &
done

wait 

# evaluation
for expid in 1 2 3 4 5 7 10; do
    nohup julia crosswalk_eval.jl --single --policy=single$expid/policy_single.bson --logdir=single$expid > eval_single$expid.$HOSTNAME.out &
    nohup julia crosswalk_eval.jl --policy=multi$expid/policy.bson --logdir=multi$expid > eval_multi$expid.$HOSTNAME.out &
    nohup julia crosswalk_eval.jl --policy=corr$expid/policy_correction.bson --correction=single$expid/policy_single.bson --logdir=corr$expid > eval_corr$expid.$HOSTNAME.out &
done
