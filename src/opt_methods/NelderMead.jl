export NelderMead

mutable struct NelderMead{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    iterations::Int
    initial_points::Vector{Vector{T}}
end

"""
$TYPEDSIGNATURES

If an `Int` is provided the method will sample that many random initial_points and use them as initial
points for Optim NelderMead method. If a `Vector{Vector{T}}` is provided it will use them as
initial points for Optim NelderMead method.
"""
function NelderMead(
    model::ScoreDrivenModel{D, T}, n_initial_points::Int;
    f_tol::T = T(1e-6), g_tol::T = T(1e-6),
    iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6
) where {D, T}
    initial_points = create_initial_points(model, n_initial_points, LB, UB)

    NelderMead{T}(f_tol, g_tol, iterations, initial_points)
end

function NelderMead(
    model::ScoreDrivenModel{D, T}, initial_points::Vector{Vector{T}};
    f_tol::T = T(1e-6), g_tol::T = T(1e-6),
    iterations::Int = 10^5
) where {D, T}
    ensure_seeds_dimensions(model, initial_points)

    NelderMead{T}(f_tol, g_tol, iterations, initial_points)
end

optimize(func::Optim.TwiceDifferentiable, opt_method::NelderMead, verbose::Int, i::Int, time_limit_sec::Int) =
    optimize(func, opt_method, Optim.NelderMead(), verbose, i, time_limit_sec)
