

mutable struct Trajectory 
    mu
    sa
    muhat
    sahat
    ud
    psi
    epsi
    wt
end


abstract type PerceptualModel end
abstract type ResponseModel end
function Base.iterate(m::PerceptualModel, state = 1)
    fs = fieldnames(typeof(m))
    if state > length(fs)
        return nothing
    else
        return (getfield(m, fs[state]), state+1)
    end
end
function Base.iterate(m::ResponseModel, state = 1)
    fs = fieldnames(typeof(m))
    if state > length(fs)
        return nothing
    else
        return (getfield(m, fs[state]), state+1)
    end
end
#
# evaluate model
# - if any value is a distribution -> sample 
#
function draw(m::Union{PerceptualModel, ResponseModel})
    θ = [m...]
    [draw(th) for th in θ]
end
function draw(m::Array) draw.(m) end
function draw(m::Sampleable) rand(m) end
function draw(m) m end

# hold both
struct Model 
    pm::PerceptualModel
    rm::ResponseModel
end

function draw(m::Model)
    Model(typeof(m.pm)(draw(m.pm)...),
          typeof(m.rm)(draw(m.rm)...))
end

# sim from model
function (m::Model)(u)
    traj = m.pm(u)
    ysim = m.rm(traj)
end
# eval model
function (m::Model)(u, y)
    traj = m.pm(u)
    ll = m.rm(traj, y)
    sum(ll)
end

function make_models(pms::Vector{T1}, rms::Vector{T2}) where T1 <: PerceptualModel where T2 <: ResponseModel
    #
    # We need to use deepcopy to make sure that
    # changing one model does not affect the others.
    #
    [Model(deepcopy(pm), deepcopy(rm)) for rm in rms, pm in pms]
end

Base.@kwdef struct Options
    niter = 10
end

