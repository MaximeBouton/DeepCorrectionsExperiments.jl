using Revise
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
using StatsBase
using CSV
using FileIO
using BSON
using ArgParse

include("decomposed_policy.jl")
includet("value_decomposition_network.jl")

rng = MersenneTwister(1)
# rng = Random.GLOBAL_RNG

const K = 4

pomdp = OCPOMDP(ΔT = 0.5, p_birth = 0.3, max_peds = 10, γ=0.99, no_ped_prob = 0.3)

env = KMarkovEnvironment(pomdp, k=K)

model = ValueDecompositionNetwork(pomdp,
                                      Chain(),
                                      Chain(Dense(4*K,32, relu), Dense(32,32,relu), Dense(32,32,relu), Dense(32, n_actions(env))), # value branch
                                      Chain()) # advantage

solver = DeepQLearningSolver(qnetwork = model, 
                             learning_rate = 1e-4,
                             max_steps = 100_000,
                             eps_end = 0.01,
                             eps_fraction = 0.5, # -0.4/9*(parsed_args["eps_fraction"] - 1) + 0.5,
                             max_episode_length = 100,
                             target_update_freq = 5000,
                             buffer_size = 400_000,
                             train_freq = 4,
                             train_start = 50_000,
                             prioritized_replay = true,
                             prioritized_replay_alpha = 0.7,
                             prioritized_replay_epsilon = 1e-3,
                             eval_freq = 10_000,
                             save_freq = 5_000,
                             log_freq = 1000,
                             logdir = "vdn",
                             double_q = true,
                             verbose=true,
                             dueling = false)

# run solver 
policy = solve(solver, env)
                