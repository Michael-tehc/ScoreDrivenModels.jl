export cross_validation

"""
$TYPEDEF

$TYPEDFIELDS
"""
struct CrossValidation{T<:AbstractFloat}
    "Absolute errors: `abs(y_true[t] - y_pred[t])`"
    abs_errors::Matrix{T}
    "Squared errors: `(y_true[t] - y_pred[t])^2`"
    sqr_errors::Matrix{T}
    "Overall Mean Absolute Error of the forecast"
    mae::Vector{T}
    "Overall Mean Squared Error of the forecast"
    mse::Vector{T}
    crps_scores::Matrix{T} # What's this??
    mean_crps::Vector{T}

    function CrossValidation(n::Int, steps_ahead::Int)
        abs_errors = Matrix{Float64}(undef, steps_ahead, n)
        sqr_errors = similar(abs_errors)
        crps_scores = Matrix{Float64}(undef, steps_ahead, n)
        mae = Vector{Float64}(undef, steps_ahead)
        mse = similar(mae)
        mean_crps = Vector{Float64}(undef, steps_ahead)
        return new{Float64}(abs_errors, sqr_errors, mae, mse, crps_scores, mean_crps)
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

evaluate_abs_error(y::AbstractVector{T}, forecast::AbstractVector{T}) where T = abs.(y - forecast)
evaluate_sqr_error(y::AbstractVector{T}, forecast::AbstractVector{T}) where T = evaluate_abs_error(y, forecast) .^2

function evaluate_crps(y::AbstractVector{T}, scenarios::AbstractMatrix{T}) where {T}
    crps_scores = Vector{T}(undef, length(y))
    for k = 1:length(y)
        crps_scores[k] = crps(y[k], scenarios[k, :])
    end
    return crps_scores
end

"""
$TYPEDSIGNATURES

Rolling window forecast of the _mean_ of observed series `y`.

- `gas`: uninitialized (not yet fitted) model.
- `y`: input time-series.
- `forecast_length`: length of the out-of-sample, same as `ForecastLength` in GAS package in R.
- `steps_ahead`: forecast the series `steps_ahead` steps ahead.
- `S`: number of simulations to run for each forecast step.
"""
function cross_validation(
    gas::ScoreDrivenModel{<:Distribution, T}, y::Vector{T}, forecast_length::Int, steps_ahead::Int=1;
    S::Int = 10_000, initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
    opt_method = NelderMead(gas, DEFAULT_NUM_SEEDS), verbose::Integer=0
)::CrossValidation where T
    @assert forecast_length > 1
    @assert steps_ahead >= 1

    check_model_estimated(gas) && throw(ArgumentError("Model must have some unestimated parameters!"))

    start_idx = length(y) - forecast_length
    num_mle = length(y) - start_idx - steps_ahead + 1
    b = CrossValidation(num_mle, steps_ahead)
    for t in 1:num_mle
        (verbose >= 1) && @info "CrossValidation: step $t of $num_mle"
        gas_to_fit = deepcopy(gas)
        y_to_fit = y[t:start_idx - 1 + t] # window moves as `t` increases
        y_to_verify = y[start_idx + t:start_idx - 1 + t + steps_ahead]
        fit!(gas_to_fit, y_to_fit; initial_params, opt_method, verbose)

        forec::Forecast = initial_params !== DEFAULT_INITIAL_PARAM ?
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S, initial_params) :
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S)

        b.abs_errors[:, t] = evaluate_abs_error(y_to_verify, forec.observation_forecast)
        b.sqr_errors[:, t] = evaluate_sqr_error(y_to_verify, forec.observation_forecast)
        b.crps_scores[:, t] = evaluate_crps(y_to_verify, forec.observation_scenarios)
    end

    @views for i in 1:steps_ahead
        b.mae[i] = mean(b.abs_errors[i, :])
        b.mse[i] = mean(b.sqr_errors[i, :])
        b.mean_crps[i] = mean(b.crps_scores[i, :])
    end
    return b
end

"""
$TYPEDSIGNATURES

Rolling window forecast of the _variance_ of the observed series `y`.
True variance is given by `realized_variance`.
"""
function cross_validation(
    gas::ScoreDrivenModel{<:Distribution, T}, y::Vector{T}, realized_variance::Vector{T},
    forecast_length::Int, steps_ahead::Int=1;
    S::Int = 10_000, initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
    opt_method = NelderMead(gas, DEFAULT_NUM_SEEDS), verbose::Integer=0
)::CrossValidation where T
    @assert forecast_length > 1
    @assert steps_ahead >= 1
    @assert length(y) == length(realized_variance)

    check_model_estimated(gas) && throw(ArgumentError("Model must have some unestimated parameters!"))

    start_idx = length(y) - forecast_length
    # `num_mle` chosen such that last index of `variance_to_verify` equals `length(y)`:
    # Find `t` such that `start_idx - 1 + t + steps_ahead == length(y)`
    num_mle = length(y) - start_idx - steps_ahead + 1
    b = CrossValidation(num_mle, steps_ahead)
    for t in 1:num_mle
        (verbose >= 1) && @info "CrossValidation: step $t of $num_mle"
        gas_to_fit = deepcopy(gas)
        y_to_fit = y[t:start_idx - 1 + t] # window moves as `t` increases
        variance_to_verify = @view realized_variance[start_idx + t:start_idx - 1 + t + steps_ahead]
        fit!(gas_to_fit, y_to_fit; initial_params, opt_method, verbose)

        forec::Forecast = initial_params !== DEFAULT_INITIAL_PARAM ?
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S, initial_params) :
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S)

        b.abs_errors[:, t] = evaluate_abs_error(variance_to_verify, forec.variance_forecast)
        b.sqr_errors[:, t] = evaluate_sqr_error(variance_to_verify, forec.variance_forecast)
    end
    
    @views for i in 1:steps_ahead
        b.mae[i] = mean(b.abs_errors[i, :])
        b.mse[i] = mean(b.sqr_errors[i, :])
    end
    return b
end