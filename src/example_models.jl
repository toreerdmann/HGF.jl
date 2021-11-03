
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
    #@assert β > 0
    states = traj.muhat[:, 1]
    prob = 1 ./ (1 .+ exp.(-β * (2 * states .- 1)))
    [rand(Bernoulli(p)) + 0 for p in prob]
end
function (m::SoftmaxBinary)(traj::Trajectory, y)
    #
    # evaluate likelihood of responses y
    #
    β = m.β
    #@assert β > 0
    x = traj.muhat[:, 1]
    ##
    ## TODO: handle/skip missing/NaN values
    ##
    prob_c = 1 ./ (1 .+ exp.(-β * (2 * x .- 1) .* (2 * y .- 1)))
    logprob = log.(prob_c)
    sum(logprob)
end
