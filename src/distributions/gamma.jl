"""
Proof somewhere 
parametrized in \\alpha and \\theta
"""
function score(y::T, ::Type{Gamma}, param::Vector{T}) where T
    return [
        log(y) - digamma(param[1]) - log(param[2]);
        y/param[2]^2 - param[1]/param[2]
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{Gamma}, param::Vector{T}) where T
    [
        -trigamma(param[1])     -1/param[2];
        -1/param[2]             -2*y/param[2]^3 + param[1]/param[2]^2
    ]
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Gamma}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik += (param[i][1] - 1)*log(y[i]) - y[i]/param[i][2] - loggamma(param[i][1]) - param[i][1]*log(param[i][2])
    end
    return -loglik
end

# Links
function link(::Type{Gamma}, param::Vector{T}) where T 
    return [
        link(LogLink, param[1], zero(T));
        link(LogLink, param[2], zero(T))
    ]
end
function unlink(::Type{Gamma}, param_tilde::Vector{T}) where T 
    return [
        unlink(LogLink, param_tilde[1], zero(T));
        unlink(LogLink, param_tilde[2], zero(T))
    ]
end
function jacobian_link(::Type{Gamma}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_link(LogLink, param_tilde[1], zero(T));
        jacobian_link(LogLink, param_tilde[2], zero(T))
    ])
end

# utils 
function update_dist(::Type{Gamma}, param::Vector{T}) where T
    return Gamma(param[1], param[2])
end 

function num_params(::Type{Gamma})
    return 2
end
