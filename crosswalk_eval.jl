using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--single"
        help = "set max_peds to 1"
        action = :store_true
    "--policy"
        help = "specify the policy to load "
        arg_type = String
        default = joinpath("log","policy.bson")
    "--logdir"
        help = "Directory in which to save the evaluation results"
        arg_type = String
        default = "log"
    "--n_eval"
        help = "Number of episodes for evaluation"
        arg_type = Int64
        default = 1000
    "--correction"
        help = "specify the single policy to load "
        arg_type = Union{Nothing, String }
        default = nothing
end
parsed_args = parse_args(s)

using Random
using POMDPs
using POMDPSimulators
using POMDPModelTools
using AutomotiveDrivingModels 
using AutomotiveSensors
using AutomotivePOMDPs
using BeliefUpdaters
using Flux
using DeepQLearning
using DeepCorrections
using FileIO
using BSON
using CSV
using StatsBase

include("decomposed_policy.jl")

const N_EVAL = parsed_args["n_eval"]
const MAX_STEPS = 100

if parsed_args["single"]
    const MAX_PEDS = 1
else
    const MAX_PEDS = 10
end

println("Evaluating in environment with $MAX_PEDS pedestrians")
pomdp = OCPOMDP(ΔT = 0.5, p_birth = 0.3, max_peds = MAX_PEDS)

println("Loading policy...")

if parsed_args["correction"] != nothing
    single_policy = BSON.load(parsed_args["correction"])[:policy]
    lowfi_policy = DecPolicy(single_policy, pomdp, (x,y) -> min.(x,y))
    correction_network = BSON.load(parsed_args["policy"])[:correction]
    problem = BSON.load(parsed_args["policy"])[:problem]
    policy = DeepCorrectionPolicy(problem, correction_network, lowfi_policy, additive_correction, 1.0, ordered_actions(pomdp))    
else
    BSON.@load parsed_args["policy"] policy
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

CSV.write(joinpath(parsed_args["logdir"], "log.csv"), df)
CSV.write(joinpath(parsed_args["logdir"], "summary.csv"), df)

println("Results saved in $(parsed_args["logdir"])")
