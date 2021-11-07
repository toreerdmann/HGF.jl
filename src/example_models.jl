
#
# Perceptual model
#
Base.@kwdef mutable struct BinaryeHGF <: PerceptualModel
    mu_0 = [NaN, 0, .1]
    sa_0 = exp.([NaN, log(1), log(1)])
    ka   = exp.([log(1), log(1)])
    om   = [NaN, -3]  
    th   = exp(-6)
end
function (m::BinaryeHGF)(u)
    # run model on inputs
    θ = draw(m)
    ehgf(u, θ...)
end

#
# Response model
#

Base.@kwdef mutable struct SoftmaxBinary <: ResponseModel
    β = 1.0
end
function (m::SoftmaxBinary)(traj::Trajectory)
    #
    # simulate responses y
    #
    β = m.β
    y = []
    trials = 1:size(traj.muhat, 1)
    for t in trials
        p = traj.muhat[t,1]
        pt = exp(β * p .- logsumexp(β .* [p, 1-p]))
        push!(y, rand(Bernoulli(pt)))
    end
    y
end
function (m::SoftmaxBinary)(traj::Trajectory, y)
    #
    # evaluate likelihood of responses y
    #
    β = m.β
    logprob = 0
    trials = setdiff(1:length(y), findall(isnan.(y)))
    for t in trials
        p = traj.muhat[t,1]
        pt = exp(β * p .- logsumexp(β .* [p, 1-p]))
        logprob += logpdf(Bernoulli(pt), y[t])
    end
    sum(logprob)
end
