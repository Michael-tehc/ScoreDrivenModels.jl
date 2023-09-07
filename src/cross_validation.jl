export cross_validation

struct CrossValidation{T<:AbstractFloat}
    abs_errors::Matrix{T}
    mae::Vector{T}
    crps_scores::Matrix{T}
    mean_crps::Vector{T}

    function CrossValidation(n::Int, steps_ahead::Int)
        abs_errors = Matrix{Float64}(undef, steps_ahead, n)
        crps_scores = Matrix{Float64}(undef, steps_ahead, n)
        mae = Vector{Float64}(undef, steps_ahead)
        mean_crps = Vector{Float64}(undef, steps_ahead)
        return new{Float64}(abs_errors, mae, crps_scores, mean_crps)
    end
end

discrete_crps_indicator_function(val::T, z::T) where {T} = val < z
function crps(val::T, scenarios::Vector{T}) where {T}
    sorted_scenarios = sort(scenarios)
    m = length(scenarios)
    crps_score = zero(T)
    for i = 1:m
        crps_score +=
            (sorted_scenarios[i] - val) *
            (m * discrete_crps_indicator_function(val, sorted_scenarios[i]) - i + 0.5)
    end
    return (2 / m^2) * crps_score
end
evaluate_abs_error(y::Vector{T}, forecast::Vector{T}) where T = abs.(y - forecast)
function evaluate_crps(y::Vector{T}, scenarios::Matrix{T}) where {T}
    crps_scores = Vector{T}(undef, length(y))
    for k = 1:length(y)
        crps_scores[k] = crps(y[k], scenarios[k, :])
    end
    return crps_scores
end

"""
$TYPEDSIGNATURES

Rolling window forecast.
"""
function cross_validation(
    gas::ScoreDrivenModel{<:Distribution, T}, y::Vector{T}, forecast_length::Int, steps_ahead::Int=1;
    S::Int = 10_000, initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
    opt_method = NelderMead(gas, DEFAULT_NUM_SEEDS), verbose=DEFAULT_VERBOSE
) where T
    @assert forecast_length > 1
    @assert steps_ahead >= 1

    check_model_estimated(gas) && throw(ArgumentError("Model must have some unestimated parameters!"))

    start_idx = length(y) - forecast_length
    num_mle = length(y) - start_idx - steps_ahead
    b = CrossValidation(num_mle, steps_ahead)
    for t in 1:num_mle
        (verbose >= 1) && @info "CrossValidation: step $t of $num_mle"
        gas_to_fit = deepcopy(gas)
        y_to_fit = y[t:start_idx - 1 + t] # window moves as `t` increases
        y_to_verify = y[start_idx + t:start_idx - 1 + t + steps_ahead]
        fit!(gas_to_fit, y_to_fit; initial_params, opt_method, verbose)

        forec::Forecast = initial_params !== DEFAULT_INITIAL_PARAM ?
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S=S, initial_params=initial_params) :
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S=S)
        abs_errors = evaluate_abs_error(y_to_verify, forec.observation_forecast)
        crps_scores = evaluate_crps(y_to_verify, forec.observation_scenarios)
        b.abs_errors[:, t] = abs_errors
        b.crps_scores[:, t] = crps_scores
    end
    for i in 1:steps_ahead
        b.mae[i] = mean(b.abs_errors[i, :])
        b.mean_crps[i] = mean(b.crps_scores[i, :])
    end
    return b
end
