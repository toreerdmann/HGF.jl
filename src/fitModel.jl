
# #
# # TODO: add MAP trajectory to output of fitModel
# # TODO: -> below code is wrong (need to use SA (i.e. SAME algorithm) or some other way)
# #
# m_MAP = deepcopy(prior)
# i_MAP = findmax(fit[3][:lp].data)[2][1]
# vals = [fit[3].value[i_MAP, p, 1]
#         for p in fit[3].name_map.parameters]
# get_parameters(m::Model) = filter(x -> x[2] isa Sampleable, [m...])
# ps = first.(
#             [
#              get_parameters(prior)
#              ...]
#            )
# for (k, v) in zip(ps, vals)
#     HGF.set_parameter!(m_MAP, k => v)
# end





function get_parameters(m::Union{PerceptualModel, ResponseModel})
    θ = [m...]
    ps = []
    for (nm, val) in zip(fieldnames(typeof(m)), θ)
        if val isa Array
            for (i,v) in enumerate(val)
                if v isa Sampleable
                    push!(ps, nm => i => v)
                end
            end
        elseif val isa Sampleable
            push!(ps, nm => val)
        end
    end
    return ps
end

get_parameters(m::Model) = filter(x -> x[2] isa Sampleable, [m...])
export get_parameters

function set_parameter!(m::Model, p::Pair)
    k, v = p
    for submodel in [m.pm, m.rm]
        if k in fieldnames(typeof(submodel))
            if v isa Pair
                f = getfield(submodel, k)
                f[v[1]] = v[2]
                setfield!(submodel, k, f)
            else
                setfield!(submodel, k, v)
            end
            break
        end
    end
end

function set_parameter!(m::Model, p::Pair, value)
    #
    # Use value instead of the leaf in pair
    #
    k, v = p
    for submodel in [m.pm, m.rm]
        if k in fieldnames(typeof(submodel))
            if v isa Pair
                f = getfield(submodel, k)
                f[v[1]] = value
                setfield!(submodel, k, f)
            else
                setfield!(submodel, k, value)
            end
            break
        end
    end
end

function pointwise_ll(chn, u, y, m, priors)
    niter = size(chn, 1)
    ll = zeros(niter)
    ll_MAP = -Inf
    m_MAP  = deepcopy(m)
    for iter in 1:niter
        for (j,th) in enumerate(chn.name_map.parameters)
            c = chn[th].data
            p = [priors[1]...; priors[2]][j]
            if p[2] isa Pair
                set_parameter!(m, p[1] => p[2][1] => c[iter])
            else
                set_parameter!(m, p[1] => c[iter])
            end
        end
        ll[iter] = m(u, y)
        # also get the maximum
        if ll[iter] > ll_MAP
            ll_MAP = ll[iter]
            m_MAP  = deepcopy(m)
        end
    end
    ll, ll_MAP, m_MAP
end

@model function post(u, y, m::Model, priors)
    pp, pr = priors
    J1 = length(pp)
    J2 = length(pr)
    θ  = zeros(Real, J1+J2)
    #
    # sample and set model parameters 
    #
    # perceptual model
    for j in 1:J1
        if pp[j][2] isa Pair
            θ[j] ~ pp[j][2][2]
            f = getfield(m.pm, pp[j][1])
            f[pp[j][2][1]] = θ[j]
            setfield!(m.pm, pp[j][1], f)
        else
            θ[j] ~ pp[j][2]
            setfield!(m.pm, pp[j][1], θ[j])
        end
    end
    # response model
    for (i,j) in enumerate(J1+1:(J1+J2))
        if pr[i][2] isa Pair
            θ[j] ~ pr[i][2][2]
            f = getfield(m.rm, pr[i][1])
            f[pr[i][2][1]] = θ[j]
            setfield!(m.rm, pr[i][1], f)
        else
            θ[j] ~ pr[i][2]
            setfield!(m.rm, pr[i][1], θ[j])
        end
    end
    # evaluate model
    try
        traj = m.pm(u)
        Turing.@addlogprob! m.rm(traj, y)
    catch e
        @show e
        if e == DomainError
            # these errors are expected
            # -> reject sample and continue
            Turing.@addlogprob! -Inf
            return
        else
            throw(e)
        end
    end
end

function fitModel(model::Model, u, y, opt::Options = Options())
    m = deepcopy(model)
    # get non-fixed parameters
    priors = get_parameters.([m.pm, m.rm])
    (length(priors) == 0) && error("No parameters to estimate!")
    if opt.sample
        # use sampling
        chn = sample(post(u, y, m, priors), NUTS(), opt.niter)
        ll, ll_MAP, m_MAP = pointwise_ll(chn, u, y, m, priors)
        loo = PSIS.psisloo(ll)
    else
        error("not implemented")
    end
    # return model with posterior distributions
    # as elements for the parameters that were given priors
    for (j,th) in enumerate(chn.name_map.parameters)
        c = chn[th].data
        p = [priors[1]...; priors[2]][j]
        if p[2] isa Pair
            approx_post = fit(typeof(p[2][2]), vec(c))
            set_parameter!(m, p[1] => p[2][1] => approx_post)
        else
            approx_post = fit(typeof(p[2]), vec(c))
            set_parameter!(m, p[1] => approx_post)
        end
    end
    return (model = m, maxll = ll_MAP, m_MAP = m_MAP, loo = loo, u = u, y = y, chn = chn)
end


# using Sobol
# ## generates points in the box [-1,1]×[0,3]×[0,2]:
# SobolSeq([-1,0,0],[1,3,2])



# priors = get_parameters.([m.pm, m.rm])
# out = optimize(HGF.post(u, y, m, priors), MLE())

# set bounds
function set_bounds(m::Model)
    ps = [get_parameters(m.pm)...; get_parameters(m.rm)...]
    function get_bounds(p::Pair)
        if p[2] isa Pair
            get_bounds(p[2])
        else
            s = support(p[2])
            [s.lb s.ub]
        end
    end
    bs = reduce(vcat, [get_bounds(p) for p in ps])
    inds = isfinite.(bs) .== false
    bs[inds] = sign.(bs[inds]) .* 10
    lb = bs[:,1]
    ub = bs[:,2]
    lb, ub
end



function optimize_model(m::Model, u, y; nstarts = 10)
    ps = [get_parameters(m.pm)...; get_parameters(m.rm)...]
    @assert length(ps) > 0
    m0 = deepcopy(m)
    f = θ -> begin
        [HGF.set_parameter!(m0, p, v) for (p,v) in zip(ps, θ)]
        try
            -m0(u, y)
        catch e
            if e isa DomainError
                return Inf
            else
                throw(e)
            end
        end
    end
    # set bounds
    lb, ub = set_bounds(m)
    # generates points in the box [-1,1]×[0,3]×[0,2]:
    s = SobolSeq(lb, ub)
    best = deepcopy(m)
    llbest = Inf
    for rep in 1:nstarts
        x0 = next!(s)
        @show x0
        try
            # out = optimize(f, lb, ub, x0,
            #                Fminbox(NelderMead()))
             out = optimize(f, lb, ub, x0, Fminbox(BFGS()),
                            autodiff = :forward)
            #out = optimize(f, lb, ub, x0, Fminbox(SimulatedAnnealing()))
            @show out.minimum, out.minimizer
            if out.minimum < llbest
                llbest = out.minimizer
                m0 = deepcopy(m)
                for (p,v) in zip(ps, θ)
                    HGF.set_parameter!(m0, p, v)
                end
                best = deepcopy(m0)
            end
        catch
        end
    end
    best, llbest
end

