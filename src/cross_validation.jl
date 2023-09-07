export cross_validation

"""
$TYPEDEF

$TYPEDFIELDS
"""
struct CrossValidation{T<:AbstractFloat}
    """
    Whether absolute & squared errors were calculated
    between `log(y_true)` and `log(y_false)` or just the raw time-series.
    """
    between_logarithms::Bool
    "Absolute errors: `abs(y_true[t] - y_pred[t])`"
    abs_errors::Matrix{T}
    "Squared errors: `(y_true[t] - y_pred[t])^2`"
    sqr_errors::Matrix{T}
    qlike_errors::Matrix{T}
    "Overall Mean Absolute Error of the forecast"
    mae::Vector{T}
    "Overall Mean Squared Error of the forecast"
    mse::Vector{T}
    "Quasi likelihood (for variance forecasts only)"
    qlike::Vector{T}
    crps_scores::Matrix{T} # What's this??
    mean_crps::Vector{T}

    function CrossValidation(between_logarithms::Bool, n::Int, steps_ahead::Int)
        abs_errors = Matrix{Float64}(undef, steps_ahead, n)
        sqr_errors = similar(abs_errors)
        qlike_errors = similar(abs_errors)
        crps_scores = Matrix{Float64}(undef, steps_ahead, n)
        mae = Vector{Float64}(undef, steps_ahead)
        mse = fill(NaN, size(mae))
        qlike = fill(NaN, size(mse))
        mean_crps = Vector{Float64}(undef, steps_ahead)
        return new{Float64}(between_logarithms, abs_errors, sqr_errors, qlike_errors, mae, mse, qlike, crps_scores, mean_crps)
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

evaluate_abs_error(y::AV{<:Real}, forecast::AV{<:Real}) = abs.(y - forecast)
evaluate_sqr_error(y::AV{<:Real}, forecast::AV{<:Real}) = evaluate_abs_error(y, forecast) .^2
evaluate_qlike_error(y::AV{<:Real}, forecast::AV{<:Real}) = @. y / forecast - log(y / forecast) - 1

function evaluate_crps(y::AV{T}, scenarios::AbstractMatrix{T}) where T<:Real
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
    gas::ScoreDrivenModel{<:Distribution, T}, y::AV{T}, forecast_length::Int, steps_ahead::Int=1;
    S::Int = 10_000, initial_params::AbstractMatrix{T} = DEFAULT_INITIAL_PARAM,
    opt_method = NelderMead(gas, DEFAULT_NUM_SEEDS), verbose::Integer=0
)::CrossValidation where T
    @assert forecast_length > 1
    @assert steps_ahead >= 1

    check_model_estimated(gas) && throw(ArgumentError("Model must have some unestimated parameters!"))

    start_idx = length(y) - forecast_length
    num_mle = length(y) - start_idx - steps_ahead + 1
    b = CrossValidation(false, num_mle, steps_ahead)
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

- `use_logarithms::Bool`: apply log transform to `realized_variance` and forecasted variance
when calculating metrics (except QLIKE).
   - `true`: metrics for `log(realized_variance)` vs `log(predicted_variance)`.
   - `false`: metrics for raw `realized_variance` vs `predicted_variance`.
"""
function cross_validation(
    gas::ScoreDrivenModel{<:Distribution, T}, y::AV{T}, realized_variance::AV{T},
    forecast_length::Int, steps_ahead::Int=1; use_logarithms::Bool=true,
    S::Int = 10_000, initial_params::AbstractMatrix{T} = DEFAULT_INITIAL_PARAM,
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
    b = CrossValidation(use_logarithms, num_mle, steps_ahead)
    for t in 1:num_mle
        (verbose >= 1) && @info "CrossValidation: step $t of $num_mle"
        gas_to_fit = deepcopy(gas)
        y_to_fit = y[t:start_idx - 1 + t] # window moves as `t` increases
        variance_to_verify = @view realized_variance[start_idx + t:start_idx - 1 + t + steps_ahead]
        fit!(gas_to_fit, y_to_fit; initial_params, opt_method, verbose)

        forec::Forecast = initial_params !== DEFAULT_INITIAL_PARAM ?
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S, initial_params) :
                    forecast(y_to_fit, gas_to_fit, steps_ahead; S)

        # Compute MAE & MSE either between raw variances
        # or between logs of variances.
        var_true, var_fc = if use_logarithms
            log.(variance_to_verify), log.(forec.variance_forecast)
        else
            variance_to_verify, forec.variance_forecast
        end
        b.abs_errors[:, t] = evaluate_abs_error(var_true, var_fc)
        b.sqr_errors[:, t] = evaluate_sqr_error(var_true, var_fc)
        # QLIKE computes logs internally
        b.qlike_errors[:, t] = evaluate_qlike_error(variance_to_verify, forec.variance_forecast)
    end
    
    @views for i in 1:steps_ahead
        b.mae[i] = mean(b.abs_errors[i, :])
        b.mse[i] = mean(b.sqr_errors[i, :])
        b.qlike[i] = mean(b.qlike_errors[i, :])
    end
    return b
end