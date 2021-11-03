
###
### julia implementation of binary HGF (enhanced version)
###

using LinearAlgebra: I

function ehgf(u, mu_0, sa_0, ka, om, th)

    # Number of levels always 3
    l = 3;

    # Unpack parameters
    # mu_0 = p[1:l];
    # sa_0 = p[l+1:2*l];
    # rho  = p[2*l+1:3*l];
    # ka   = p[3*l+1:4*l-1];
    # om   = p[4*l:5*l-2];
    # th   = exp(p[5*l-1]);

    #@assert th > 0

    # Add dummy "zeroth" trial
    u = [0; u[:,1]]
    # Number of trials (including prior)
    n = length(u);
    t = ones(n,1);

    # Initialize updated quantities

    # Representations
    mu    = zeros(Real, n,l)
    pii   = zeros(Real, n,l)
    # Other quantities
    muhat = zeros(Real, n,l);
    pihat = zeros(Real, n,l);
    v     = zeros(Real, n,l);
    w     = zeros(Real, n,l-1);
    da    = zeros(Real, n,l);

    t = ones(Real, n)

    # Representation priors
    # Note: first entries of the other quantities remain
    # NaN because they are undefined and are thrown away
    # at the end; their presence simply leads to consistent
    # trial indices.
    sgm = (x, a) -> a ./ (1 .+ exp.(-1 .* x))
    mu[1,1] = sgm(mu_0[1], 1)
    pii[1,1] = Inf
    mu[1,2:end] .= mu_0[2:end]
    pii[1,2:end] .= 1 ./ sa_0[2:end]

    # Pass through representation update loop
    for k = 2:1:n
        #%%%%%%%%%%%%%%%%%%%%%
        # Effect of input u[k]
        #%%%%%%%%%%%%%%%%%%%%%

        muhat[k,2] = mu[k-1,2]
        muhat[k,1] = sgm(ka[1] *muhat[k,2], 1)
        # fix for NaN trajectories
        if muhat[k,1] <= 0.001
            muhat[k,1] = 0.001
        elseif muhat[k,1] >= 0.999
            muhat[k,1] = 0.999
        end
        pihat[k,1] = 1/(muhat[k,1]*(1 -muhat[k,1]))
        pii[k,1] = Inf
        mu[k,1] = u[k]
        da[k,1] = mu[k,1] -muhat[k,1]

        pihat[k,2] = 1 / (1 / pii[k-1,2] + exp(ka[2] *mu[k-1,3] +om[2]))

        pii[k,2] = pihat[k,2] +ka[1]^2 / pihat[k,1]
        mu[k,2] = muhat[k,2] +ka[1]/pii[k,2] *da[k,1]

        # Volatility prediction error
        da[k,2] = (1/pii[k,2] +(mu[k,2] -muhat[k,2])^2) * pihat[k,2] -1

        ##
        ## Third level
        ##

        # Prediction
        muhat[k,l] = mu[k-1,l] 
        # Precision of prediction
        pihat[k,l] = 1/(1/pii[k-1,l] + th)
        # Weighting factor
        v[k,l]   = t[k] * th
        v[k,l-1] = t[k] * exp(ka[l-1] *mu[k-1,l] +om[l-1])
        w[k,l-1] = v[k,l-1] *pihat[k,l-1]

        # Mean update
        mu[k,l] = muhat[k,l] + 1/2 * 1/pihat[k,l] * ka[l-1] * w[k,l-1] *da[k,l-1];

        # Ingredients of the precision update which depend on the mean
        # update
        vv = t[k] * exp(ka[l-1] * mu[k,l] + om[l-1])
        pimhat = 1 / (1 / pii[k-1,l-1] + vv)
        ww = vv * pimhat
        rr = (vv - 1/pii[k-1,l-1]) * pimhat
        dd = (1/pii[k,l-1] + (mu[k,l-1] - muhat[k,l-1])^2) * pimhat - 1

        pii[k,l] = pihat[k,l] + max(0, 1/2 * ka[l-1]^2 * ww * (ww + rr * dd))

        # Volatility prediction error
        da[k,l] = (1/pii[k,l] + (mu[k,l] -muhat[k,l])^2) * pihat[k,l] - 1

    end

    # Implied learning rate at the first level
    sgmmu2 = sgm(ka[1] .* mu[:,2], 1)
    dasgmmu2 = u .- sgmmu2
    lr1    = diff(sgmmu2) ./ dasgmmu2[2:n,1]
    lr1[da[2:n,1].==0] .= 0

    # Remove other dummy initial values
    mu = mu[2:end, :]
    pii = pii[2:end, :]
    muhat = muhat[2:end, :]
    pihat = pihat[2:end, :]
    v = v[2:end, :]
    w = w[2:end, :]
    da = da[2:end, :]

    #any(isnan.(muhat)) && error("NaN muhat")
    any(isnan.(muhat)) && throw(DomainError)

    ##
    ## Create result data structure
    ##

    # Updates with respect to prediction
    ud = mu .- muhat

    # Psi (precision weights on prediction errors)
    psi        = zeros(Real, n-1, l)
    psi[:,2]   = 1 ./ pii[:,2]
    psi[:,3] = pihat[:,2] ./ pii[:,3]

    # Epsilons (precision-weighted prediction errors)
    epsi          = zeros(Real, n-1, l);
    epsi[:, 2:3] .= psi[:,2] .* da[:,1:2]

    # Full learning rate (full weights on prediction errors)
    wt        = zeros(Real, n-1, l);
    wt[:,1]   = lr1;
    wt[:,2]   = psi[:,2];
    wt[:,3]   = 1/2 .* (v[:,2:l-1] * I * ka[2:l-1]) .* psi[:,3:l]

    Trajectory(mu, 1 ./ pii, muhat, 1 ./ pihat, ud, psi, epsi, wt)
end


#function ehgf(u::Array{Int, 1}, p_prc::Parameters)
#    ehgf(u, p_prc.mu_0, p_prc.sa_0, p_prc.ka, p_prc.om, p_prc.th)
#end

#
# for real inputs, automatically use model with
# perceptual uncertainty
#
function ehgf(u, mu_0, sa_0, ka, om, th, al, eta0, eta1)

    # Number of levels always 3
    l = 3;

    # Unpack parameters
    # mu_0 = p[1:l];
    # sa_0 = p[l+1:2*l];
    # rho  = p[2*l+1:3*l];
    # ka   = p[3*l+1:4*l-1];
    # om   = p[4*l:5*l-2];
    # th   = exp(p[5*l-1]);
    # al   = p(5*l);
    # eta0 = p(5*l+1);
    # eta1 = p(5*l+2);

    #@assert th > 0

    # Add dummy "zeroth" trial
    u = [0; u[:,1]]
    # Number of trials (including prior)
    n = length(u);
    t = ones(n,1);

    # Initialize updated quantities

    # Representations
    # Other quantities
    mu    = zeros(Real, n,l)
    pii   = zeros(Real, n,l)
    muhat = zeros(Real, n,l);
    pihat = zeros(Real, n,l);
    v     = zeros(Real, n,l);
    w     = zeros(Real, n,l-1);
    da    = zeros(Real, n,l);

    t = ones(Int, n)

    # Representation priors
    # Note: first entries of the other quantities remain
    # NaN because they are undefined and are thrown away
    # at the end; their presence simply leads to consistent
    # trial indices.
    sgm = (x, a) -> a ./ (1 .+ exp.(-1 .* x))
    mu[1,1] = sgm(mu_0[1], 1)
    pii[1,1] = Inf
    mu[1,2:end] .= mu_0[2:end]
    pii[1,2:end] .= 1 ./ sa_0[2:end]

    # Pass through representation update loop
    for k = 2:1:n
        #%%%%%%%%%%%%%%%%%%%%%
        # Effect of input u[k]
        #%%%%%%%%%%%%%%%%%%%%%

        muhat[k,2] = mu[k-1,2]
        muhat[k,1] = sgm(ka[1] *muhat[k,2], 1)
        pihat[k,1] = 1/(muhat[k,1]*(1 -muhat[k,1]))
        pii[k,1] = Inf

        #mu(k,1) = u(k);
        und1 = exp(-(u[k] -eta1)^2 / (2*al))
        und0 = exp(-(u[k] -eta0)^2 / (2*al))
        mu[k,1] = muhat[k,1] * und1 / 
        (muhat[k,1] * und1 + (1 - muhat[k,1]) * und0)


        da[k,1] = mu[k,1] -muhat[k,1]

        pihat[k,2] = 1 / (1 / pii[k-1,2] + exp(ka[2] *mu[k-1,3] +om[2]))

        pii[k,2] = pihat[k,2] +ka[1]^2 / pihat[k,1]
        mu[k,2] = muhat[k,2] +ka[1]/pii[k,2] *da[k,1]

        # Volatility prediction error
        da[k,2] = (1/pii[k,2] +(mu[k,2] -muhat[k,2])^2) * pihat[k,2] -1

        ##
        ## Third level
        ##

        # Prediction
        muhat[k,l] = mu[k-1,l] 
        # Precision of prediction
        pihat[k,l] = 1/(1/pii[k-1,l] + th)
        # Weighting factor
        v[k,l]   = t[k] * th
        v[k,l-1] = t[k] * exp(ka[l-1] *mu[k-1,l] +om[l-1])
        w[k,l-1] = v[k,l-1] *pihat[k,l-1]

        # Mean update
        mu[k,l] = muhat[k,l] + 1/2 * 1/pihat[k,l] * ka[l-1] * w[k,l-1] *da[k,l-1];

        # Ingredients of the precision update which depend on the mean
        # update
        vv = t[k] * exp(ka[l-1] * mu[k,l] + om[l-1])
        pimhat = 1 / (1 / pii[k-1,l-1] + vv)
        ww = vv * pimhat
        rr = (vv - 1/pii[k-1,l-1]) * pimhat
        dd = (1/pii[k,l-1] + (mu[k,l-1] - muhat[k,l-1])^2) * pimhat - 1

        pii[k,l] = pihat[k,l] + max(0, 1/2 * ka[l-1]^2 * ww * (ww + rr * dd))

        # Volatility prediction error
        da[k,l] = (1/pii[k,l] + (mu[k,l] -muhat[k,l])^2) * pihat[k,l] - 1

    end

    # Implied learning rate at the first level
    sgmmu2 = sgm(ka[1] .* mu[:,2], 1)
    dasgmmu2 = u .- sgmmu2
    lr1    = diff(sgmmu2) ./ dasgmmu2[2:n,1]
    lr1[da[2:n,1].==0] .= 0

    # Remove other dummy initial values
    mu = mu[2:end, :]
    pii = pii[2:end, :]
    muhat = muhat[2:end, :]
    pihat = pihat[2:end, :]
    v = v[2:end, :]
    w = w[2:end, :]
    da = da[2:end, :]

    #any(isnan.(muhat)) && error("NaN muhat")
    any(isnan.(muhat)) && throw(DomainError)

    ##
    ## Create result data structure
    ##

    # Updates with respect to prediction
    ud = mu .- muhat

    # Psi (precision weights on prediction errors)
    psi        = zeros(Real, n-1, l)
    psi[:,2]   = 1 ./ pii[:,2]
    psi[:,3] = pihat[:,2] ./ pii[:,3]

    # Epsilons (precision-weighted prediction errors)
    epsi          = zeros(Real, n-1, l);
    epsi[:, 2:3] .= psi[:,2] .* da[:,1:2]

    # Full learning rate (full weights on prediction errors)
    wt        = zeros(Real, n-1, l);
    wt[:,1]   = lr1;
    wt[:,2]   = psi[:,2];
    wt[:,3]   = 1/2 .* (v[:,2:l-1] * I * ka[2:l-1]) .* psi[:,3:l]

    Trajectory(mu, 1 ./ pii, muhat, 1 ./ pihat, ud, psi, epsi, wt)
end

# function ehgf(u::Array{Real, 1}, p_prc::Parameters)
#     ehgf(u, p_prc.mu_0, p_prc.sa_0, p_prc.ka, p_prc.om, p_prc.th, 
#          p_prc.al, p_prc.eta0, p_prc.eta1)
# end

