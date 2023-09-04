# HGF.jl

This repo holds a minimal implementation of the Hierarchical Gaussian Filter
(HGF) for analysing behavioral data in julia. 

__Warning: This is untested and experimental research software__, please do not
use it unless you know what you are doing.
 

## Installation

You can install the package with the following command (from the julia REPL):

```julia
        ]add https://github.com/toreerdmann/HGF.jl
```

## Usage

```julia
using HGF
using Distributions: Normal, Exponential
using Plots

# draw design
ps = repeat([.2, .8, .2], inner = 100)
u  = [rand() < p for p in ps]

# Define a model
M = Model(BinaryeHGF(mu_0 = [NaN, 1, Exponential(1)],
                     sa_0 = [NaN, .1, .1],
                     om   = [NaN, Normal(-5, 1)],
                     th   = .1),
          SoftmaxBinary())

# Draw a model from the prior and run it forward
m = draw(M)
traj = m.pm(u)
plt = scatter(u .+ randn(length(u)) .* .01, legend=:none)
plot!(plt, traj.muhat[:,1]) 

# draw subjects
nsubs = 10
subs = [draw(M) for rep in 1:nsubs]
# simulate from each subject
ysim = [m(u) for m in subs]

# fit data
fits = [fitModel(M, u, ysim[i],
                 Options(niter = 200))[1] for i in 1:nsubs]

# look at simulated parameter vs. estimates 
omtrue = [sub.pm.om[2] for sub in subs]
omest  = [fit.pm.om[2].μ for fit in fits]
omest_err  = [fit.pm.om[2].σ for fit in fits]
scatter(omtrue, omest, yerror=omest_err)
plot!(x -> x)



```

