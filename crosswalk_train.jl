using Printf
using Random
using POMDPs
using POMDPPolicies
using POMDPSimulators
using BeliefUpdaters
using AutomotiveDrivingModels
using AutomotiveSensors
using AutomotivePOMDPs
using AutoViz
using Flux
using RLInterface
using DeepQLearning
using DeepCorrections
using FileIO
using BSON
using ArgParse

include("decomposed_policy.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--single"
        help = "set max_peds to 1"
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
    "--logdir"
        help = "Directory in which to save the model and log training data"
        arg_type = String
        default = "log"
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

const K = 4

pomdp = OCPOMDP(ΔT = 0.5, p_birth = 0.3, max_peds = MAX_PEDS, γ=0.99, no_ped_prob = 0.3)

env = KMarkovEnvironment(pomdp, k=K)

input_dims = reduce(*, obs_dimensions(env))
model = Chain(x->flattenbatch(x), Dense(input_dims, 32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32,n_actions(env), relu))
dqn_solver = DeepQLearningSolver(qnetwork = model, 
                             learning_rate = 1e-4,
                             max_steps = parsed_args["training_steps"],
                             eps_end = parsed_args["eps_end"],
                             eps_fraction = parsed_args["eps_fraction"],
                             max_episode_length = 100,
                             target_update_freq = 5000,
                             buffer_size = 400_000,
                             train_freq = 4,
                             train_start = 10_000,
                             prioritized_replay = true,
                             prioritized_replay_alpha = 0.7,
                             prioritized_replay_epsilon = 1e-3,
                             eval_freq = 10_000,
                             save_freq = 10_000,
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
                                  lowfi_values = lowfi_policy)
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

