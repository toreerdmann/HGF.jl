
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

Base.@kwdef mutable struct Parameters 
    mu_0
    sa_0
    ka
    om
    th
    nu
end

function pars(p::Parameters)
    [p.mu_0, p.sa_0, p.ka, p.om, p.th, p.nu]
end

function parsv(p::Parameters)
    vcat([p.mu_0, p.sa_0, p.ka, p.om, p.th, p.nu]...)
end


## take priors from hgf toolbox and transform
mu_0 = [NaN, 0, 1]
sa_0 = exp.([NaN, log(.1), log(1)])
ka   = exp.([log(1), log(1)])
om   = [NaN, -4]
th   = exp(-6)

## random exploration of parameter space
function draw_parameters(n)
    params = []
    for i in 1:n
        # fix mu01 because it make sense
        mu_0 = [NaN; 0; rand(Uniform(-10, 10))]
        # fix sa01 and sa02, because ....
        sa_0 = [NaN; .1; 1]
        # fix ka because of unidentifiability
        # ka = rand(Uniform(-10, 10), 2)
        ka = [1, 1]
        om = [NaN; rand(Uniform(-10, 10))]
        th = rand(Uniform(-10, 10))
        nu = rand(Uniform(-10, 10))
        push!(params, Parameters(mu_0, sa_0, ka, om, th, nu))
    end
    params
end
