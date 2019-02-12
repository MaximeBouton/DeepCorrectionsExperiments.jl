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
using RLInterface
using Flux
using DeepQLearning
using BSON

rng = MersenneTwister(1)

const K = 4

pomdp = OCPOMDP(ΔT = 0.5, p_birth = 0.3, max_peds = 10, no_ped_prob= 0.0)

env = KMarkovEnvironment(pomdp, k=K)

up = KMarkovUpdater(K)
# helpers for KMarkovUpdater
POMDPs.convert_o(::Type{Array{Float64}}, o::Array{Array{Float64, 1}}, pomdp::OCPOMDP) = hcat(o...)

correction = false
if correction
    single_policy = BSON.load(parsed_args["correction"])[:policy]
    lowfi_policy = DecPolicy(single_policy, pomdp, (x,y) -> min.(x,y))
    correction_network = BSON.load(parsed_args["policy"])[:correction]
    problem = BSON.load(parsed_args["policy"])[:problem]
    policy = DeepCorrectionPolicy(problem, correction_network, lowfi_policy, additive_correction, 1.0, ordered_actions(pomdp))    
else
    BSON.@load  "log/policy.bson" policy
end

# policy = RandomPolicy(pomdp, rng=rng)


hr = HistoryRecorder(rng=rng, max_steps = 100)
s0 = initialstate(pomdp, rng)
initial_observation = generate_o(pomdp, s0, rng)
initial_obs_vec = fill(initial_observation, K)
hist = simulate(hr, pomdp, policy, up, initial_obs_vec, s0)



# Visualize
using Reel
frames = Frames(MIME("image/png"), fps=4)
AutoViz.render(s0, pomdp.env, cam = StaticCamera(VecE2(25.0,0.0), 15.0))
for step in eachstep(hist, "s,a,r,sp")
    s, a, r, sp = step
    push!(frames, AutoViz.render(sp, pomdp.env, cam=StaticCamera(VecE2(25.0,0.0), 15.0)))
end
write("out.gif", frames)


simlist = []
for i=1:100
    rng = MersenneTwister(i)
    s0 = initialstate(pomdp, rng)
    o0 = generate_o(pomdp, s0, rng)
    b0 = initialize_belief(up, fill(o0, K))
    push!(simlist, Sim(pomdp, policy, up, b0, s0, rng=rng, max_steps=100))
end

println("Starting Parallel Simulation...")

df = run_parallel(simlist) do sim, hist
    return (n_steps=n_steps(hist), 
            obs_hist = hist.observation_hist,
            reward=discounted_reward(hist), 
            collision=undiscounted_reward(hist) < 0.0, 
            timeout=undiscounted_reward(hist)==0.0, 
            success=undiscounted_reward(hist)>0.0)
end

using StatsBase
summary = describe(df, stats=[:mean, :std])

DeepQLearning.basic_evaluation(policy,KMarkovEnvironment(pomdp, k=K), 100, 100, true)

function DeepQLearning.basic_evaluation(policy::AbstractNNPolicy, env::AbstractEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool)
    avg_r = 0 
    for i=1:n_eval
        done = false 
        r_tot = 0.0
        step = 0
        obs = reset!(env)
        resetstate!(policy)
        while !done && step <= max_episode_length
            act = action(policy, obs)
            obs, rew, done, info = step!(env, act)
            @show obs
            @show rew, step
            r_tot += rew 
            step += 1
        end
        avg_r += r_tot 
    end
    if verbose
        println("Evaluation ... Avg Reward ", avg_r/n_eval)
    end
    return  avg_r /= n_eval
end


obs = df[:obs_hist][1][1]

actionvalues(policy, obs)

o = update(up, initial_obs_vec, OCAction(0.0), obs)

o_vec = 

actionvalues(policy, o)