


function get_parameters(m)
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

function pointwise_ll(chn, u, y, m, priors)
    niter = size(chn, 1)
    ll = zeros(niter)
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
    end
    ll
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
    chn = sample(post(u, y, m, priors), NUTS(), opt.niter)
    # return model with posterior distributions
    # as elements for some parameters
    ll = pointwise_ll(chn, u, y, m, priors)
    loo = PSIS.psisloo(ll)
    # construct approximate posterior
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
    return m, [maximum(ll), loo], chn
end


