struct ValueDecompositionNetwork{B, V, A} <: AbstractNNPolicy
    pomdp::OCPOMDP
    base::B
    val::V
    adv::A
end

function (m::ValueDecompositionNetwork)(inpt)
    decomposed_input = [get_singlestate(m.pomdp, inpt, i) for i in 1:m.pomdp.max_peds] 
    x = flattenbatch.(decomposed_input) # array of arrays 4*K x bs
    out = m.val(x[1]) # 4xbs
    for i=2:length(x)
        out += m.val(x[i])
    end
    out /= m.pomdp.max_peds

    # d, bs = size(first(x))
    # x = Flux.batch(x) # matrix inp x bs x agents
    # x = reshape(x, d, :) 
    # out = m.val(x) #.+ m.adv(x) .- mean(m.adv(x), dims=1)
    # out = reshape(out, n_actions(m.pomdp), pomdp.max_peds, :)

    # # fusion 
    # out = mean(out, dims=2) # sum over number of pedestrians , dim n_a x bs
    # out = reshape(out, (n_actions(m.pomdp), bs))



    # x = m.base(inpt) #m.base(inpt)
    # ndim, bs = size(x)
    # local_dim = div(ndim, m.n)
    # loc_x = x[1:local_dim, :]
    # # out = m.val(loc_x) .+ m.adv(loc_x) .- mean(m.adv(loc_x), dims=1)
    # out = m.adv(loc_x)
    # for i=1:m.n-1
    #     loc_x = x[i*local_dim + 1: i*local_dim + local_dim, :]
    #     # out += m.val(loc_x) .+ m.adv(loc_x) .- mean(m.adv(loc_x), dims=1)
    #     out += m.adv(loc_x)
    # end
    return out
end

function Flux.params(m::ValueDecompositionNetwork)
    ps = Flux.Params()
    Flux.prefor(p ->
    Tracker.istracked(p) && Tracker.isleaf(p) &&
        !any(p′ -> p′ === p, ps) && push!(ps, p),
    m.base)
    Flux.prefor(p ->
    Tracker.istracked(p) && Tracker.isleaf(p) &&
        !any(p′ -> p′ === p, ps) && push!(ps, p),
    m.adv)
    Flux.prefor(p ->
    Tracker.istracked(p) && Tracker.isleaf(p) &&
        !any(p′ -> p′ === p, ps) && push!(ps, p),
    m.val)
    return ps
end

function Flux.reset!(m::ValueDecompositionNetwork)
    Flux.reset!(m.base)
end

function Base.deepcopy(m::ValueDecompositionNetwork)
  ValueDecompositionNetwork(deepcopy(m.pomdp), deepcopy(m.base), deepcopy(m.val), deepcopy(m.adv))
end

function Base.iterate(m::ValueDecompositionNetwork, i=1)
    if i > length(m.base.layers) + length(m.val.layers) + length(m.adv.layers)
        return nothing 
    end
    if i <= length(m.base.layers)
        return (m.base[i], i+1)
    elseif i <= length(m.base.layers) + length(m.val.layers)
        return (m.val[i - length(m.base.layers)], i+1)
    elseif i <= length(m.base.layers) + length(m.val.layers) + length(m.adv.layers)
        return (m.adv[i - length(m.base.layers) - length(m.val.layers)], i+1)
    end   
end