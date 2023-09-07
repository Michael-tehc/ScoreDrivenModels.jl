function check_model_estimated(len::Int)::Bool
    if len == 0
        @warn "Score Driven Model does not have unknowns."
        return true
    end
    return false
end

check_model_estimated(gas::ScoreDrivenModel)::Bool =
    gas |> find_unknowns |> length |> check_model_estimated

function NaN2zero!(m::Matrix{T}, i::Integer) where T
    for j in axes(m, 2)
        if isnan(m[i, j])
            m[i, j] = zero(T) 
        end
    end
    return 
end

function big_threshold!(m::Matrix{T}, threshold::F, i::Integer) where {T, F}
    for j in axes(m, 2)
        if m[i, j] >= threshold
            m[i, j] = threshold 
        end
        if m[i, j] <= -threshold
            m[i, j] = -threshold 
        end
    end
    return 
end

function small_threshold!(m::Matrix{T}, threshold::F, i::Integer) where {T, F}
    for j in axes(m, 2)
        if m[i, j] <= threshold && m[i, j] >= 0
            m[i, j] = threshold 
        end
        if m[i, j] >= -threshold && m[i, j] <= 0
            m[i, j] = -threshold 
        end
    end
    return 
end

function sample_observation(dist::Distribution)
    return rand(dist)
end

function find_unknowns(v::Vector{T}) where T
    return findall(isnan, v)
end

function find_unknowns(m::Matrix{T}) where T
    return findall(isnan, vec(m))
end