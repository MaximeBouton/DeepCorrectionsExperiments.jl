using Printf
using Random
using POMDPs
using POMDPPolicies
using POMDPSimulators
using BeliefUpdaters
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutoViz
using Flux
using RLInterface
using DeepQLearning
using DeepCorrections
using StatsBase
using CSV
using FileIO
using BSON
using ArgParse

include("decomposed_policy.jl")
include("value_decomposition_network.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--single"
        help = "set max_peds to 1"
        action = :store_true
    "--vdn"
        help = "value decomposition network"
        action = :store_true
    "--seed"
        help = "specify the random seed"
        arg_type = Int
        default = 1
    "--training_steps"
        help = "specify the number of training steps"
        arg_type = Int
        default = 10_000
    "--eps_fraction"
        help = "Fraction of the training set to use to decay epsilon from 1.0 to eps_end"
        arg_type = Float64
        default = 0.5
    "--eps_end"
        help = "epsilon value at the end of the exploration phase"
        arg_type = Float64
        default = 0.01
    "--correction"
        help = "specify the single policy to load "
        arg_type = Union{Nothing, String }
        default = nothing
    "--weight"
        help = "weight of the correction network"
        arg_type = Float64
        default = 0.1
    "--logdir"
        help = "Directory in which to save the model and log training data"
        arg_type = String
        default = "log"
    "--n_eval"
        help = "Number of episodes for evaluation"
        arg_type = Int64
        default = 1000
end
parsed_args = parse_args(s)

if parsed_args["single"]
    MAX_PEDS = 1
else
    MAX_PEDS = 10
end

seed = parsed_args["seed"]
Random.seed!(seed)
rng = MersenneTwister(seed)
# rng = Random.GLOBAL_RNG

const K = 4

pomdp = OCPOMDP(ΔT = 0.5, p_birth = 0.3, max_peds = MAX_PEDS, γ=0.99, no_ped_prob = 0.3)

env = KMarkovEnvironment(pomdp, k=K)

input_dims = reduce(*, obs_dimensions(env))
if parsed_args["correction"] != nothing 
    model = Chain(x->flattenbatch(x), Dense(input_dims, 32, relu), Dense(32,32,relu), Dense(32, n_actions(env)))
elseif parsed_args["single"]
    model = Chain(x->flattenbatch(x), Dense(input_dims, 32, relu), Dense(32,32,relu), Dense(32, n_actions(env)))
elseif parsed_args["vdn"]
    model = ValueDecompositionNetwork(pomdp,
                                      Chain(Dense(4*K,64, relu), Dense(64,32,relu), Dense(32,32,relu), Dense(32, n_actions(env)))) 
else
    model = Chain(x->flattenbatch(x), Dense(input_dims, 32, relu), Dense(32,32, relu),  Dense(32,32, relu),  Dense(32,32, relu), Dense(32,n_actions(env)))
end
dqn_solver = DeepQLearningSolver(qnetwork = model, 
                             learning_rate = 1e-4,
                             max_steps = parsed_args["training_steps"],
                             eps_end = parsed_args["eps_end"],
                             eps_fraction = parsed_args["eps_fraction"], # -0.4/9*(parsed_args["eps_fraction"] - 1) + 0.5,
                             max_episode_length = 100,
                             target_update_freq = 5000,
                             buffer_size = 400_000,
                             train_freq = 4,
                             train_start = 10_000,
                             prioritized_replay = true,
                             prioritized_replay_alpha = 0.7,
                             prioritized_replay_epsilon = 1e-3,
                             eval_freq = 10_000,
                             save_freq = 5_000,
                             log_freq = 1000,
                             logdir = parsed_args["logdir"],
                             double_q = true,
                             verbose=true)

if parsed_args["correction"] == nothing 
    solver = dqn_solver
else
    single_policy = BSON.load(parsed_args["correction"])[:policy]
    lowfi_policy = DecPolicy(single_policy, pomdp, (x,y) -> min.(x,y))
    solver = DeepCorrectionSolver(dqn = dqn_solver,
                                  lowfi_values = lowfi_policy,
                                  correction_weight = parsed_args["weight"])
end

if parsed_args["vdn"]
    solver.dueling = false 
end

# run solver 
policy = solve(solver, env)

# save policy 
isdir(parsed_args["logdir"]) ? nothing : mkdir(parsed_args["logdir"]) 

policy_name = "policy"
if parsed_args["single"]
    policy_name *= "_single.bson"
elseif parsed_args["correction"] != nothing
    policy_name *= "_correction.bson"
else
    policy_name *= ".bson"
end

if parsed_args["correction"] != nothing
    BSON.bson(joinpath(parsed_args["logdir"], policy_name), Dict(:correction => policy.correction_network, :problem => policy.problem))
else
    BSON.@save joinpath(parsed_args["logdir"], policy_name) policy
end

## eval policy 

const N_EVAL = parsed_args["n_eval"]
const MAX_STEPS = 100

println("Evaluating in environment with 10 pedestrians")
pomdp = OCPOMDP(ΔT = 0.5, p_birth = 0.3, max_peds = 10, no_ped_prob = 0.1)

if parsed_args["single"]
    policy =  DecPolicy(policy, pomdp, (x,y) -> min.(x,y))
    println("Initialized Decomposed Policy")
end


const K = 4
updater = KMarkovUpdater(K)

# helpers for KMarkovUpdater
POMDPs.convert_o(::Type{Array{Float64}}, o::Array{Array{Float64, 1}}, pomdp::OCPOMDP) = hcat(o...)

simlist = []
for i=1:N_EVAL
    rng = MersenneTwister(i)
    s0 = initialstate(pomdp, rng)
    o0 = generate_o(pomdp, s0, rng)
    b0 = initialize_belief(updater, fill(o0, K))
    push!(simlist, Sim(pomdp, policy, updater, b0, s0, rng=rng, max_steps=MAX_STEPS))
end

println("Starting Parallel Simulation...")

df = run_parallel(simlist) do sim, hist
    return (n_steps=n_steps(hist), 
            reward=discounted_reward(hist), 
            collision=undiscounted_reward(hist) < 0.0, 
            timeout=undiscounted_reward(hist)==0.0, 
            success=undiscounted_reward(hist)>0.0)
end


summary = describe(df, stats=[:mean, :std])

println(summary)

append = false
if isfile(joinpath(parsed_args["logdir"], "summary.csv"))
    append = true 
end
CSV.write(joinpath(parsed_args["logdir"], "log.csv"), df, append=append)
CSV.write(joinpath(parsed_args["logdir"], "summary.csv"), summary, append=append)

println("Results saved in $(parsed_args["logdir"])")
