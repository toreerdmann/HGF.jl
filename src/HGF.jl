module HGF

using Distributions: Sampleable, Normal, Bernoulli, Exponential, truncated
using Turing
using LogExpFunctions: logistic, logsumexp

include("core.jl")
export Options, Model, Trajectory, make_models
export PerceptualModel, ResponseModel, draw

include("fitModel.jl")
export fitModel

include("hgf_binary.jl")
include("ehgf_binary.jl")
export hgf, ehgf

include("example_models.jl")
export BinaryeHGF, SoftmaxBinary

include("PSIS.jl")
export psisloo

end
