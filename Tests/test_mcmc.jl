
#
# Try HMC inference (not working yet)
#
# -> MH is working well though
#

include("types.jl")
include("hgf_binary.jl")
include("ehgf_binary.jl")

using Turing

## read in data from tutorial
f = open("example_binary_input.txt");
lines = readlines(f)
u = parse.(Int, lines[1:end-1])

function softmax_binary(u::Array{Int, 1}, y::Array{Int, 1}, traj::Trajectory, be::Float64)
    # be = exp(be)
    x = traj.muhat[:, 1]
    probc = 1 ./ (1 .+ exp.(-be * (2 * x .- 1) .* (2 * y .- 1)))
    logp = log.(probc)
    # yhat = y .* probc + (1 .- y) .* (1 .- probc)
    # res = (y .- yh) ./ sqrt.(yh .* (1 .- yh))
    logp
end
function softmax_binary(u::Array{Int, 1}, traj::Trajectory, be::Float64)
    be = exp(be)
    states = traj.muhat[:, 1]
    prob = 1 ./ (1 .+ exp.(-be * (2 * states .- 1)))
    [rand(Distributions.Bernoulli(p)) + 0 for p in prob]
end

## simulate responses
function simModel(u)
    mu_0 = [NaN, 0, 1]
    sa_0 = exp.([NaN, log(1), log(1)])
    ## rho = [NaN, 0, 0]
    ka   = exp.([log(1), log(1)])
    om   = [NaN, -2.5]
    th   = exp(-6)
    be   = 1.0
    traj = hgf(u, mu_0, sa_0, ka, om, th)
    ysim = softmax_binary(u, traj, be)
    ysim, traj
end

y, traj = simModel(u)

scatter([u y])
plot!(traj.muhat)

ll = softmax_binary(u, y, traj, 1.0)
sum(ll)

@model function loglik(u, y, perceptual_model = hgf)
    # define priors over parameters to be inferred
    om2 ~ Normal(-5, 3) 
    #th  ~ LogNormal(-10, 5)
    th  ~ Exponential(.01)
    be   ~ Normal(0, 1)
    # fixed parameters
    mu_0 = [NaN, 0, 1]
    sa_0 = exp.([NaN, log(1), log(1)])
    ka   = exp.([log(1), log(1)])
    om   = [NaN, om2]
    th   = th
    #be   = 1.0
    try
        traj = perceptual_model(u, mu_0, sa_0, ka, om, th)
        try
            ll = softmax_binary(u, y, traj, be)
            Turing.@addlogprob! sum(ll)
            return
        catch 
            Turing.@addlogprob! -Inf
            return
        end
    catch
        Turing.@addlogprob! -Inf
        return
    end
end

# this works
chn = sample(loglik(u, y, ehgf), MH(), 100_000)
plot(chn)

using StatsPlots
# show density
density(chn.value.data[:,3])
# true value
vline!([exp(-6)])

density(chn.value.data[:,2])
vline!([-2.5])


# sample 
# -> probably better to first run optimization to find starting values
chn = sample(loglik(u, y, ehgf), NUTS(5, .65), 10)


chn = sample(loglik(u, y, ehgf), HMC(.001, 10), 10)


chn = sample(loglik(u, y, ehgf), Prior(), 5)
chn = optimize(loglik(u, y, ehgf), MLE())


