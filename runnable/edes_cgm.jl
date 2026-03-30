# edes_cgm.jl
# CGM scenario: estimate k1, k5, k6 from interstitial glucose data only (k8 fixed).
# Run from the repo root: julia runnable/edes_cgm.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.resolve(); Pkg.instantiate()
using OrdinaryDiffEq, CairoMakie, Random, QuasiMonteCarlo
using Optimization, OptimizationOptimJL, Distributions
import OptimizationOptimJL: LBFGS
include(joinpath(@__DIR__, "..", "utils", "SelectionMethods.jl"))

# ─── User-editable inputs ─────────────────────────────────────────────────────
CGM_time    = [0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125]
CGM_glu     = [6.1 6.2 6.4 6.7 7.1 7.6 8.1 8.5 8.9 9.2 9.4 9.5 9.6 9.6 9.5 9.4 9.3 9.2 9.0 8.9 8.8 8.6 8.5 8.3 8.2 7.9]
fasting_ins = 7.9    # fasting plasma insulin (uIU/ml)
k8_fixed    = 7.25   # fixed value for k8

# ─── Derived from data ────────────────────────────────────────────────────────
# CGM insulin row: fasting value at t=0, zeros elsewhere (ODE solver needs Ib to initialise)
CGM_ins  = [fasting_ins zeros(1, length(CGM_time) - 1)]
CGM_data = [CGM_time; CGM_glu; CGM_ins]

Gb = CGM_glu[1]     # fasting glucose (mmol/l)
Ib = fasting_ins    # fasting insulin (uIU/ml)

# ─── Fixed physiological parameters ──────────────────────────────────────────
bw    = 70.0
Dmeal = 75.0e3
k2    = 0.28
k3    = 6.07e-3
k4    = 2.35e-4
k7    = 1.15
k9    = 3.83e-2
k10   = 2.84e-1
tau_i = 31.0
tau_d = 3.0
beta  = 1.0
Gren  = 9.0
EGPb  = 0.043
Km    = 13.2
f     = 0.005551
Vg    = 17.0 / bw
c1    = 0.1
sigma = 1.4

constants = [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]

# ─── ODE definition ───────────────────────────────────────────────────────────
function edesode!(du, u, p, t)
    Ggut, Gpl, Ipl, Irem = u
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib = p

    du[1] = sigma * k1^sigma * t^(sigma-1) * exp(-(k1*t)^sigma) * Dmeal - k2 * Ggut

    gliv  = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
    ggut  = k2 * (f / (Vg * bw)) * Ggut
    u_ii  = EGPb * ((Km + Gb)/Gb) * (Gpl / (Km + Gpl))
    u_id  = k5 * beta * Irem * (Gpl / (Km + Gpl))
    u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)
    du[2] = gliv + ggut - u_ii - u_id - u_ren

    i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * Gb + k8 * tau_d * du[2])
    i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
    i_int = k9 * (Ipl - Ib)
    du[3] = i_pnc - i_liv - i_int
    du[4] = i_int - k10 * Irem
end

# ─── construct_parameters: θ = [k1, k5, k6], k8 fixed ───────────────────────
function construct_parameters(θ, c)
    k1 = θ[1]; k5 = θ[2]; k6 = θ[3]; k8 = k8_fixed
    k2 = c[1]; k3 = c[2]; k4 = c[3]; k7 = c[4]; k9 = c[5]; k10 = c[6]
    tau_i = c[7]; tau_d = c[8]; beta = c[9]; Gren = c[10]; EGPb = c[11]
    Km = c[12]; f = c[13]; Vg = c[14]; c1 = c[15]; sigma = c[16]
    Dmeal = c[17]; bw = c[18]; Gb = c[19]; Ib = c[20]
    return [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

# ─── ODE problem ─────────────────────────────────────────────────────────────
u0    = [0.0, Gb, Ib, 0.0]
tspan = (0.0, 240.0)
prob  = ODEProblem(edesode!, u0, tspan, constants)

# ─── CGM loss function (glucose only, θ = [k1,k5,k6,σ_g,σ_i]) ───────────────
# θ[end-1] = σ_g is used; θ[end] = σ_i is carried in the parameter vector
# to share bounds with the 3-param OGTT estimation but is not used in the loss.
const first_error_cgm = Ref(true)
function loss_CGM(θ, (problem, constants, data))
    glucose_data    = data[2,:]
    data_timepoints = data[1,:]
    penalty = sum(abs2, θ[1:3]) * 1e4 + 1e6   # finite, has gradient → LBFGS recovers

    p = construct_parameters(θ, constants)
    local pred
    try
        pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2],
                     u0=[0.0, data[2,1], data[3,1], 0.0])
    catch e
        if first_error_cgm[]
            println("Loss function error on ODE solve: ", e)
            first_error_cgm[] = false
        end
        return penalty
    end

    sol = vec(Array(pred))
    if any(isnan, sol)
        if first_error_cgm[]
            println("Solution contains NaN with θ = ", θ)
            first_error_cgm[] = false
        end
        return penalty
    end
    length(sol) == length(glucose_data) || return penalty
    g_loss = (sol - glucose_data) / maximum(glucose_data)
    any(!isfinite, g_loss)              && return penalty

    n   = length(glucose_data)
    σ_g = θ[end-1]
    L_g = n*log(σ_g * sqrt(2π)) + 1/(2σ_g^2) * sum(abs2, g_loss)
    isfinite(L_g) || return penalty
    return 0.5 * L_g
end

# ─── Initial guess generation ─────────────────────────────────────────────────
function generate_initial_guesses(n_guesses, lower_bounds, upper_bounds; selection_method::SelectionMethod = SelectAll())
    initial_guesses = QuasiMonteCarlo.sample(n_guesses, lower_bounds, upper_bounds, LatinHypercubeSample())
    return evaluate_and_select(initial_guesses, selection_method)
end

# ─── Parameter estimation ─────────────────────────────────────────────────────
lb = [0.0, 0.0, 0.0, 1e-4, 1e-4]
ub = [1.0, 0.5, 10.0, 100.0, 100.0]
initial_guess = generate_initial_guesses(10, lb, ub)
println("Initial guess size: ", size(initial_guess))
println("Initial guess sample: ", initial_guess[:, 1:min(3, end)])
println("Contains NaN: ", any(isnan, initial_guess))

results = []
optf = OptimizationFunction(loss_CGM, AutoForwardDiff())

for (i, guess) in enumerate(eachcol(initial_guess))
    if i == 1
        # First iteration: no try-catch to see full error
        res = solve(OptimizationProblem(optf, Vector(guess), (prob, constants, CGM_data), lb=lb, ub=ub), LBFGS())
        push!(results, res)
    else
        try
            res = solve(OptimizationProblem(optf, Vector(guess), (prob, constants, CGM_data), lb=lb, ub=ub), LBFGS())
            push!(results, res)
        catch e
            continue
        end
    end
end
println("CGM estimation: $(length(results))/1000 runs succeeded")
if isempty(results)
    error("All optimisation runs failed — remove try-catch to diagnose.")
end

best_index = argmin([r.objective for r in results])
final = results[best_index].u

solution = solve(prob, p=construct_parameters(final, constants), u0=[0.0, CGM_data[2,1], fasting_ins, 0.0])

# ─── Figure: CGM model fit ────────────────────────────────────────────────────
fig_fit = let f = Figure(size=(500,500))
    ax_g_gut    = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]",       title="Gut Glucose")
    ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]",  title="Plasma Glucose")
    ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
    ax_i_int    = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

    lines!(ax_g_gut,    solution.t, solution[1,:], color=Makie.wong_colors()[1])
    lines!(ax_g_plasma, solution.t, solution[2,:], color=Makie.wong_colors()[1])
    lines!(ax_i_plasma, solution.t, solution[3,:], color=Makie.wong_colors()[1])
    lines!(ax_i_int,    solution.t, solution[4,:], color=Makie.wong_colors()[1])

    band!(ax_g_plasma, solution.t, solution[2,:] .+ final[end-1], solution[2,:] .- final[end-1])
    band!(ax_i_plasma, solution.t, solution[3,:] .+ final[end],   solution[3,:] .- final[end])

    scatter!(ax_g_plasma, CGM_data[1,:], CGM_data[2,:], color=Makie.wong_colors()[2])
    f
end
save(joinpath(@__DIR__, "fig_cgm_fit.png"), fig_fit)

println("Final estimated parameters (CGM fit): ", final)

# ─── PLA infrastructure ───────────────────────────────────────────────────────
struct PLAResult
    likelihood_values
    parameter_values
    optim_index::Int
    other_parameter_values
end

function loss_known_sigma_CGM(θ, (problem, constants, data, sigma_g, sigma_i))
    glucose_data    = data[2,:]
    data_timepoints = data[1,:]
    penalty = sum(abs2, θ[1:3]) * 1e4 + 1e6

    p = construct_parameters(θ, constants)
    local pred
    try
        pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2],
                     u0=[0.0, data[2,1], data[3,1], 0.0])
    catch
        return penalty
    end

    sol = vec(Array(pred))
    length(sol) == length(glucose_data) || return penalty
    g_loss = (sol - glucose_data) / maximum(glucose_data)
    any(!isfinite, g_loss)              && return penalty

    n   = length(glucose_data)
    L_g = n*log(sigma_g * sqrt(2π)) + 1/(2sigma_g^2) * sum(abs2, g_loss)
    return 0.5 * L_g
end

function loss_pla(loss, pla_param, fixed_parameter_value, npar)
    parameter_order = zeros(Int64, npar)
    parameter_order[[1:pla_param-1; (pla_param+1):npar]] .= 1:npar-1
    parameter_order[pla_param] = npar
    function _loss_pla(θ, p)
        θ_full = [θ; fixed_parameter_value][parameter_order]
        loss(θ_full, p)
    end
end

function pla_step(pla_param_index, loss, θ_init, lb, ub, p)
    npar   = length(θ_init)
    lb_pla = [lb[1:pla_param_index-1]; lb[(pla_param_index+1):npar]]
    ub_pla = [ub[1:pla_param_index-1]; ub[(pla_param_index+1):npar]]
    initial_guess = [θ_init[1:pla_param_index-1]; θ_init[pla_param_index+1:end]]
    function _likelihood(pla_param)
        loss_function = loss_pla(loss, pla_param_index, pla_param, length(initial_guess)+1)
        optfunc = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
        optprob = OptimizationProblem(optfunc, initial_guess, p, lb=lb_pla, ub=ub_pla)
        optsol  = solve(optprob, LBFGS())
        optsol.objective, optsol.u
    end
end

function run_pla(pla_param, param_range, param_optim, loss, initial_guess, lb_pla, ub_pla, p)
    step = pla_step(pla_param, loss, initial_guess, lb_pla, ub_pla, p)
    param_range = sort(unique([param_range; param_optim]))
    optim_index = findfirst(x -> x == param_optim, param_range)
    pla_likelihood_values  = Float64[]
    other_parameter_values = typeof(initial_guess)[]
    parameter_values       = typeof(param_optim)[]
    for px in param_range
        try
            l, _p = step(px)
            push!(pla_likelihood_values, l)
            push!(other_parameter_values, _p)
            push!(parameter_values, px)
        catch e
            throw(e)
        end
    end
    return PLAResult(pla_likelihood_values, parameter_values, optim_index, other_parameter_values)
end

function likelihood_profile!(ax, pla_result::PLAResult, param_index, param_name)
    parameter_values  = pla_result.parameter_values
    likelihood_values = pla_result.likelihood_values .- minimum(pla_result.likelihood_values)
    lines!(ax, parameter_values, likelihood_values, color=Makie.wong_colors()[1])
end

# ─── Run PLA ──────────────────────────────────────────────────────────────────
ranges = [
    range(1e-5, 0.02, length=800),
    range(0.001, 0.2, length=800),
    range(0.5, 20.0, length=400),
]

pla_results = [
    run_pla(i, ranges[i], final[i], loss_known_sigma_CGM, final[1:3],
            [0.0, 0.0, 0.0], [1.0, 0.5, 10.0],
            (prob, constants, CGM_data, final[end-1], final[end]))
    for i in 1:3
]

# ─── Figure: PLA ──────────────────────────────────────────────────────────────
fig_pla = let f = Figure()
    positions   = [(1,1), (1,2), (2,1)]
    param_names = ["k1", "k5", "k6"]
    for (i, (pos, param_name)) in enumerate(zip(positions, param_names))
        ax = Axis(getindex(f, pos...), xlabel=param_name, ylabel="ΔL", title="Profile Likelihood for $param_name")
        likelihood_profile!(ax, pla_results[i], i, param_name)
        scatter!(ax, [final[i]], [0], color=Makie.wong_colors()[2])
        hlines!(ax, [quantile(Chisq(1), 0.95)], linestyle=:dash, color=Makie.wong_colors()[3])
        ylims!(ax, (0.0, 5.0))
    end
    f
end
save(joinpath(@__DIR__, "fig_cgm_pla.png"), fig_pla)

println("Done. Figures saved: fig_cgm_fit.png, fig_cgm_pla.png")
