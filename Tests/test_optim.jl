
## Test my implementation
using LinearAlgebra
using Distributions
using Random
using Plots
using StatsPlots
using Optim

include("types.jl")
include("hgf_binary.jl")
include("ehgf_binary.jl")

## read in data from tutorial
# -> this did not match the output from matlab below (because its different inputs)
# f = open("u.csv");
# lines = readlines(f)
# u = parse.(Int, lines)

# read in 
f = open("example_binary_input.txt");
lines = readlines(f)
u = parse.(Int, lines[1:end-1])

## define priors as 
## take priors from hgf toolbox and transform
mu_0 = [NaN, 0, 1]
sa_0 = exp.([NaN, log(1), log(1)])
## rho = [NaN, 0, 0]
ka   = exp.([log(1), log(1)])
om   = [NaN, -2.5]
th   = exp(-6)

traj = hgf(u, mu_0, sa_0, ka, om, th)
plot(traj.muhat)
scatter!(u)

traj2 = ehgf(u, mu_0, sa_0, ka, om, th)
plot!(traj2.muhat)


## output matches results in tutorial function:
traj.muhat[1:5, :]
# 5×3 Array{Float64,2}:
#  0.5       0.0       1.0     
#  0.614993  0.468351  0.995083
#  0.693902  0.818426  0.984989
#  0.749732  1.09718   0.973582
#  0.790869  1.33017   0.962363
## in matlab:
# u = load('example_binary_input.txt');
# sim = tapas_simModel(u,...
#                      'tapas_hgf_binary',...
#                      [NaN 0 1 NaN 1 1 NaN 0 0 1 1 NaN -2.5 -6],...
#                      'tapas_unitsq_sgm',...
#                      5,...
#                      12345);
# sim.muhat
   # 0.5000         0    1.0000
   #  0.6150    0.4684    0.9951
   #  0.6939    0.8184    0.9850
   #  0.7497    1.0972    0.9736
   #  0.7909    1.3302    0.9624

##
## Now try fit
##
# bopars = tapas_fitModel([],...
#                          u,...
#                          'tapas_hgf_binary_config',...
#                          'tapas_bayes_optimal_binary_config',...
#                          'tapas_quasinewton_optim_config');


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
    traj = hgf(u, mu_0, sa_0, ka, om, th)
    ysim = softmax_binary(u, traj, 1.0)
    ysim, traj
end

y, traj = simModel(u)

scatter([u y])
plot!(traj.muhat)

ll = softmax_binary(u, y, traj, 1.0)
sum(ll)

## and fit them
function fitModel(u, y, f = hgf)
    npars = 2
    ## define loss function
    ## (specify which parameter to optimise through x)
    function loglik(x::Array{Float64, 1}, y, u)
        mu_0 = [NaN, 0, 1]
        sa_0 = exp.([NaN, log(1), log(1)])
        ## rho = [NaN, 0, 0]
        ka   = exp.([log(1), log(1)])
        # om   = [NaN, -2.5]
        # th   = exp(-6)
        om   = [NaN, x[1]]
        th   = exp(x[2])
        be   = 1.0
        try
            traj = f(u, mu_0, sa_0, ka, om, th)
            try
                ll = softmax_binary(u, y, traj, be)
                return -1 * sum(ll)
            catch 
                return Inf
            end
        catch
            return Inf
        end
    end
    ## run optimiser
    @time opt = optimize(x -> loglik(x, y, u), rand(Uniform(-20, 10), npars))
    opt
end

opts = [fitModel(u, y) for rep in 1:20];

## look at ll values at minimum
[o.minimum for o in opts]
[o.minimizer for o in opts if o.minimum != Inf]

## compare to random
x = repeat([.5], length(y));
-1 * sum(log.(1 ./ (1 .+ exp.(-1 * (2 * x .- 1) .* (2 * y .- 1)))))
x = rand(length(y));
-1 * sum(log.(1 ./ (1 .+ exp.(-1 * (2 * x .- 1) .* (2 * y .- 1)))))

##
## do the same with ehgf
##

nstarts = 20
opts = [fitModel(u, y, ehgf) for rep in 1:nstarts];
scores    = [o.minimum for o in opts]
# filter out Infs
opts = [o for o in opts if o.minimum != Inf]
## look at ll values at minimum
scores    = [o.minimum for o in opts]
estimates = [o.minimizer for o in opts]
## compare to random
x = repeat([.5], length(y));
-1 * sum(log.(1 ./ (1 .+ exp.(-1 * (2 * x .- 1) .* (2 * y .- 1)))))
x = rand(length(y));
-1 * sum(log.(1 ./ (1 .+ exp.(-1 * (2 * x .- 1) .* (2 * y .- 1)))))

##
## -> more re-starts needed to get a stable solution
##

nstarts = 100
out = map(1:5) do rep
    opts = [fitModel(u, y, ehgf) for rep in 1:nstarts];
    scores    = [o.minimum for o in opts]
    # filter out Infs
    opts = [o for o in opts if o.minimum != Inf]
    ## look at ll values at minimum
    scores    = [o.minimum for o in opts]
    estimates = [o.minimizer for o in opts]
    # compare parameter estimates of best 3 solutions
    [scores[sortperm(scores)[1:5]] estimates[sortperm(scores)[1:5]]]
end


