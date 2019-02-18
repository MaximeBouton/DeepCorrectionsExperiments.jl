#!/bin/bash
for expid in 1 2 3 4 5 7 10; do
    nohup julia crosswalk_eval.jl --single --policy=single$expid/policy_single.bson --logdir=single$expid > eval_single$expid.$HOSTNAME.out &
    nohup julia crosswalk_eval.jl --policy=multi$expid/policy.bson --logdir=multi$expid > eval_multi$expid.$HOSTNAME.out &
    nohup julia crosswalk_eval.jl --policy=corr$expid/policy_correction.bson --correction=single$expid/policy_single.bson --logdir=corr$expid > eval_corr$expid.$HOSTNAME.out &
done
