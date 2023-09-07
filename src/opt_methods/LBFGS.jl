export LBFGS

mutable struct LBFGS{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    iterations::Int
    initial_points::Vector{Vector{T}}
end

"""
$TYPEDSIGNATURES

If an `Int` is provided the method will sample that many random initial_points and use them as
initial points for Optim LBFGS method. If a `Vector{Vector{T}}` is provided it will use them as
initial points for Optim LBFGS method.
"""
function LBFGS(
    model::ScoreDrivenModel{D, T}, n_initial_points::Int;
    f_tol::T = T(1e-6), g_tol::T = T(1e-6),
    iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6
) where {D, T}
    initial_points = create_initial_points(model, n_initial_points, LB, UB)

    LBFGS{T}(f_tol, g_tol, iterations, initial_points)
end

function LBFGS(
    model::ScoreDrivenModel{D, T}, initial_points::Vector{Vector{T}};
    f_tol::T = T(1e-6), g_tol::T = T(1e-6),
    iterations::Int = 10^5
) where {D, T}
    ensure_seeds_dimensions(model, initial_points)

    LBFGS{T}(f_tol, g_tol, iterations, initial_points)
end

optimize(func::Optim.TwiceDifferentiable, opt_method::LBFGS, verbose::Int, i::Int, time_limit_sec::Int) =
    optimize(func, opt_method, Optim.LBFGS(), verbose, i, time_limit_sec)

