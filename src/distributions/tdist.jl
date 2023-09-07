"""
    Student's t

* Parametrization
parametrized in \\nu

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
TDist

function score!(score_til::Matrix{T}, y::T, ::Type{TDist}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = 0.5 * (((y^2)*(param[t, 1] + 1)/(param[t, 1] * y^2 + param[t, 1]^2)) -
                      1/param[t, 1] - 
                      log((y^2)/(param[t, 1]) + 1) - 
                      digamma(param[t, 1] / 2) + digamma((param[t, 1] + 1) / 2))
    return
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{TDist}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = 0.5 * (0.5 * trigamma(0.5 * param[t, 1]) - 0.5 * trigamma( 0.5 * (param[t, 1] + 1.0) ) -
                             (param[t, 1] + 5.0) / (param[t, 1] * (param[t, 1] + 3.0) * (param[t, 1] + 1.0)))
    return
end

"""
$TYPEDSIGNATURES

Actually returns _negative_ log-likelihood.
"""
function log_likelihood(::Type{TDist}, y::Vector{T}, param::Matrix{T}, nobs::Int)::T where T
    loglik = zero(T)
    for t in 1:nobs
        loglik -= 0.5 * log(param[t, 1]) + logbeta(0.5, param[t, 1]/2) + ((param[t, 1] + 1)/2) *
                  log(1 + (y[t]^2)/param[t, 1])
    end
    -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{TDist}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{TDist}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{TDist}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    return
end

# utils 
update_dist(::Type{TDist}, param::Matrix, t::Int) =
    TDist(param[t, 1])

params_sdm(d::TDist) = Distributions.params(d)

num_params(::Type{TDist}) = 1
