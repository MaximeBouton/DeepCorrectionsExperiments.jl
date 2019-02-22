using POMDPs
using POMDPPolicies 
using AutomotivePOMDPs
using Flux

struct DecPolicy{P <: Policy, M <: Union{MDP, POMDP}, O} <: Policy 
    policy::P
    problem::M 
    op::O # reduction operator 
end

function POMDPs.action(policy::DecPolicy, s::OCObs)
    ai = argmax(actionvalues(policy, s))
    return actions(policy.problem)[ai]
end

function POMDPs.action(policy::DecPolicy, s::Vector{OCObs}) # for the KMarkov updater
    s_ = hcat(s...)
    ai = argmax(actionvalues(policy, s_))
    return actions(policy.problem)[ai]
end

function POMDPPolicies.actionvalues(policy::DecPolicy, s)
    return Flux.data(_actionvalues(policy, decompose_state(policy.problem, s)))
end

function _actionvalues(policy::DecPolicy, s_dec::AbstractArray)   # no hidden state!
    return reduce(policy.op, actionvalues(policy.policy, s) for s in s_dec)
end

function decompose_state(pomdp::OCPOMDP, s)
    return [get_singlestate(pomdp, s, i) for i in 1:pomdp.max_peds]
end

function get_singlestate(pomdp::OCPOMDP, s, i::Int) #XXX Beware of batch size!
    n_features = 2
    ego = view(s, Base.setindex(axes(s), 1:n_features, 1)...)
    ped = view(s, Base.setindex(axes(s), n_features*i + 1:n_features*(i + 1), 1)...)
    return vcat(ego, ped)
end