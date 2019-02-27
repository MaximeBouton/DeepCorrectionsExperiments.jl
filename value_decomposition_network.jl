struct ValueDecompositionNetwork{N} <: AbstractNNPolicy
    pomdp::OCPOMDP
    net::N
end

function (m::ValueDecompositionNetwork)(inpt)
    decomposed_input = [get_singlestate(m.pomdp, inpt, i) for i in 1:m.pomdp.max_peds] 
    x = flattenbatch.(decomposed_input) # array of arrays 4*K x bs
    out = m.net(x[1]) # 4xbs
    for i=2:length(x)
        out += m.net(x[i])
    end
    out /= m.pomdp.max_peds
    return out
end

function Flux.params(m::ValueDecompositionNetwork)
    ps = Flux.Params()
    Flux.prefor(p ->
    Tracker.istracked(p) && Tracker.isleaf(p) &&
        !any(p′ -> p′ === p, ps) && push!(ps, p),
    m.net)
    return ps
end

function Flux.reset!(m::ValueDecompositionNetwork)
    Flux.reset!(m.net)
end

function Base.deepcopy(m::ValueDecompositionNetwork)
  ValueDecompositionNetwork(deepcopy(m.pomdp), deepcopy(m.net))
end

function Base.iterate(m::ValueDecompositionNetwork, i=1)
    if i > length(m.net.layers)
        return nothing 
    else
        return (m.net[i], i+1)
    end
end