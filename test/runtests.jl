using HGF
using Test

# my dependency
using Distributions: Normal, Exponential
using Statistics

@testset "HGF.jl" begin
end

@testset "simple recovery" begin
    #
    # define some models
    # (see example_models.jl)
    #
    pm1 = BinaryeHGF(om = [NaN, Normal(-3, 2)])
    pm2 = BinaryeHGF(om = [NaN, Normal(-3, 2)], 
                     th = Exponential(.1))
    rm1 = SoftmaxBinary()
    rm2 = SoftmaxBinary(β = Exponential(1))
    models = make_models([pm1, pm2], [rm1, rm2])
    m = models[1]

    # draw design
    ps = repeat([.2, .8, .2], inner = 100)
    u = [rand() < p for p in ps]

    # draw subjects
    nsubs = 10
    subs = [draw(m) for rep in 1:nsubs]

    # simulate from each subject
    ysim = [m(u) for m in subs]

    # fit data
    fits = [fitModel(m, u, ysim[i], Options(niter = 100))[1] for i in 1:nsubs]

    #
    # do a check that the fits are reasonable
    #
    omtrue = [sub.pm.om[2] for sub in subs]
    omest  = [fit.pm.om[2] for fit in fits]
    @test cor(omtrue, omest) > .5

end
