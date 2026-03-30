# edes_ogtt_4param.jl
# OGTT scenario: estimate k1, k5, k6, k8 from plasma glucose + insulin data.
# Run from the repo root: julia runnable/edes_ogtt_4param.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.resolve(); Pkg.instantiate()
using OrdinaryDiffEq, CairoMakie, Random, QuasiMonteCarlo
using Optimization, OptimizationOptimJL, Distributions
import OptimizationOptimJL: LBFGS
include(joinpath(@__DIR__, "..", "utils", "SelectionMethods.jl"))

# ─── User-editable inputs ─────────────────────────────────────────────────────
data_time = [0 15 30 45 60 90 120]
data_glu  = [5.4 7.1 8.6 8.9 8.5 7.6 6.6]
data_ins  = [7.9 36.7 64.8 75.6 79.6 80.5 68.7]

# ─── Derived from data ────────────────────────────────────────────────────────
Gb = data_glu[1]   # fasting glucose (mmol/l)
Ib = data_ins[1]   # fasting insulin (uIU/ml)
data = [data_time; data_glu; data_ins]

# ─── Fixed physiological parameters ──────────────────────────────────────────
bw    = 70.0       # body weight (kg)
Dmeal = 75.0e3     # amount of glucose in meal (mg)
k2    = 0.28       # rate of glucose transport from stomach to plasma
k3    = 6.07e-3    # rate of glucose effect on endogenous glucose production
k4    = 2.35e-4    # rate of insulin effect on endogenous glucose production
k7    = 1.15       # rate of insulin secretion (integral of Gpl)
k9    = 3.83e-2    # delay parameter plasma to interstitial insulin
k10   = 2.84e-1    # rate constant for degradation of insulin in remote compartment
tau_i = 31.0       # time delay integrator function
tau_d = 3.0        # time delay
beta  = 1.0        # conversion factor insulin
Gren  = 9.0        # threshold for renal excretion of glucose
EGPb  = 0.043      # basal rate of endogenous glucose production
Km    = 13.2       # Michaelis-Menten coefficient for glucose uptake into periphery
f     = 0.005551
Vg    = 17.0 / bw  # volume of distribution for glucose
c1    = 0.1        # model constant
sigma = 1.4        # shape factor (appearance of meal)

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

# ─── construct_parameters: θ = [k1, k5, k6, k8] ─────────────────────────────
function construct_parameters(θ, c)
    k1 = θ[1]; k5 = θ[2]; k6 = θ[3]; k8 = θ[4]
    k2 = c[1]; k3 = c[2]; k4 = c[3]; k7 = c[4]; k9 = c[5]; k10 = c[6]
    tau_i = c[7]; tau_d = c[8]; beta = c[9]; Gren = c[10]; EGPb = c[11]
    Km = c[12]; f = c[13]; Vg = c[14]; c1 = c[15]; sigma = c[16]
    Dmeal = c[17]; bw = c[18]; Gb = c[19]; Ib = c[20]
    return [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

# ─── ODE problem ─────────────────────────────────────────────────────────────
u0   = [0.0, Gb, Ib, 0.0]
tspan = (0.0, 240.0)
prob = ODEProblem(edesode!, u0, tspan, constants)  # placeholder p; overridden in solve

# ─── Loss function (glucose + insulin, θ = [k1,k5,k6,k8,σ_g,σ_i]) ───────────
function loss(θ, (problem, constants, data))
    glucose_data   = data[2,:]
    insulin_data   = data[3,:]
    data_timepoints = data[1,:]

    p    = construct_parameters(θ, constants)
    pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2,3],
                 u0=[0.0, data[2,1], data[3,1], 0.0])
    sol  = Array(pred)

    g_loss = (sol[1,:] - glucose_data) / maximum(glucose_data)
    i_loss = (sol[2,:] - insulin_data) / maximum(insulin_data)

    n   = length(glucose_data)
    σ_g = θ[end-1]
    σ_i = θ[end]
    L_g = n*log(σ_g * sqrt(2π)) + 1/(2σ_g^2) * sum(abs2, g_loss)
    L_i = n*log(σ_i * sqrt(2π)) + 1/(2σ_i^2) * sum(abs2, i_loss)
    return 0.5 * L_g + 0.5 * L_i
end

# ─── Initial guess generation ─────────────────────────────────────────────────
function generate_initial_guesses(n_guesses, lower_bounds, upper_bounds; selection_method::SelectionMethod = SelectAll())
    initial_guesses = QuasiMonteCarlo.sample(n_guesses, lower_bounds, upper_bounds, LatinHypercubeSample())
    return evaluate_and_select(initial_guesses, selection_method)
end

# ─── Parameter estimation ─────────────────────────────────────────────────────
lb = [0.0, 0.0, 0.0, 0.0, 1e-4, 1e-4]
ub = [1.0, 0.5, 10.0, 25.0, 100.0, 100.0]
initial_guess = generate_initial_guesses(1000, lb, ub)
println("Initial guess size: ", size(initial_guess))
println("Initial guess sample: ", initial_guess[:, 1:min(3, end)])
println("Contains NaN: ", any(isnan, initial_guess))

results = []
optf = OptimizationFunction(loss, AutoForwardDiff())

for (i, guess) in enumerate(eachcol(initial_guess))
    if i == 1
        # First iteration: no try-catch to see full error
        res = solve(OptimizationProblem(optf, Vector(guess), (prob, constants, data), lb=lb, ub=ub), LBFGS())
        push!(results, res)
    else
        try
            res = solve(OptimizationProblem(optf, Vector(guess), (prob, constants, data), lb=lb, ub=ub), LBFGS())
            push!(results, res)
        catch e
            continue
        end
    end
end
println("4-param estimation: $(length(results))/1000 runs succeeded")
if isempty(results)
    error("All optimisation runs failed — remove try-catch to diagnose.")
end

best_index = argmin([r.objective for r in results])
final = results[best_index].u

solution = solve(prob, p=construct_parameters(final, constants), u0=[0.0, data[2,1], data[3,1], 0.0])

# ─── Figure: raw OGTT data ────────────────────────────────────────────────────
fig_data = let f = Figure(size=(400,200))
    ax_glu = Axis(f[1,1], xlabel="Time [min]", ylabel="Concentration [mM]",   title="Glucose Data")
    ax_ins = Axis(f[1,2], xlabel="Time [min]", ylabel="Concentration [mU/L]", title="Insulin Data")
    scatter!(ax_glu, data[1,:], data[2,:], color=Makie.wong_colors()[1])
    scatter!(ax_ins, data[1,:], data[3,:], color=Makie.wong_colors()[1])
    f
end
save(joinpath(@__DIR__, "fig_ogtt4_data.png"), fig_data)

# ─── Figure: model fit ────────────────────────────────────────────────────────
fig_fit = let f = Figure(size=(500,500))
    ax_g_gut    = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]",      title="Gut Glucose")
    ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
    ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
    ax_i_int    = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

    lines!(ax_g_gut,    solution.t, solution[1,:], color=Makie.wong_colors()[1])
    lines!(ax_g_plasma, solution.t, solution[2,:], color=Makie.wong_colors()[1])
    lines!(ax_i_plasma, solution.t, solution[3,:], color=Makie.wong_colors()[1])
    lines!(ax_i_int,    solution.t, solution[4,:], color=Makie.wong_colors()[1])

    band!(ax_g_plasma, solution.t, solution[2,:] .+ final[5], solution[2,:] .- final[5])
    band!(ax_i_plasma, solution.t, solution[3,:] .+ final[6], solution[3,:] .- final[6])

    scatter!(ax_g_plasma, data[1,:], data[2,:], color=Makie.wong_colors()[2])
    scatter!(ax_i_plasma, data[1,:], data[3,:], color=Makie.wong_colors()[2])
    f
end
save(joinpath(@__DIR__, "fig_ogtt4_fit.png"), fig_fit)

# ─── Figure: parameter histograms (first 100 results) ────────────────────────
k1_vals = [results[r].u[1] for r in 1:min(100, length(results))]
k5_vals = [results[r].u[2] for r in 1:min(100, length(results))]
k6_vals = [results[r].u[3] for r in 1:min(100, length(results))]
k8_vals = [results[r].u[4] for r in 1:min(100, length(results))]

fig_hist = let f = Figure(size=(500,500))
    ax_k1 = Axis(f[1,1], xlabel="k1 value", title="k1 distribution")
    ax_k5 = Axis(f[1,2], xlabel="k5 value", title="k5 distribution")
    ax_k6 = Axis(f[2,1], xlabel="k6 value", title="k6 distribution")
    ax_k8 = Axis(f[2,2], xlabel="k8 value", title="k8 distribution")
    hist!(ax_k1, k1_vals, bins=10, color=Makie.wong_colors()[1])
    hist!(ax_k5, k5_vals, bins=10, color=Makie.wong_colors()[1])
    hist!(ax_k6, k6_vals, bins=10, color=Makie.wong_colors()[1])
    hist!(ax_k8, k8_vals, bins=10, color=Makie.wong_colors()[1])
    f
end
save(joinpath(@__DIR__, "fig_ogtt4_hist.png"), fig_hist)

# ─── PLA infrastructure ───────────────────────────────────────────────────────
struct PLAResult
    likelihood_values
    parameter_values
    optim_index::Int
    other_parameter_values
end

function loss_known_sigma(θ, (problem, constants, data, sigma_g, sigma_i))
    glucose_data    = data[2,:]
    insulin_data    = data[3,:]
    data_timepoints = data[1,:]

    p    = construct_parameters(θ, constants)
    pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2,3],
                 u0=[0.0, data[2,1], data[3,1], 0.0])
    sol  = Array(pred)

    g_loss = (sol[1,:] - glucose_data) / maximum(glucose_data)
    i_loss = (sol[2,:] - insulin_data) / maximum(insulin_data)

    n   = length(glucose_data)
    L_g = n*log(sigma_g * sqrt(2π)) + 1/(2sigma_g^2) * sum(abs2, g_loss)
    L_i = n*log(sigma_i * sqrt(2π)) + 1/(2sigma_i^2) * sum(abs2, i_loss)
    return 0.5 * L_g + 0.5 * L_i
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
    range(1e-5, 0.05, length=800),
    range(0.001, 0.5, length=800),
    range(0.5, 6.0,  length=400),
    range(3.0, 12.0, length=400),
]

pla_results = [
    run_pla(i, ranges[i], final[i], loss_known_sigma, final[1:4],
            [0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 10.0, 25.0],
            (prob, constants, data, final[5], final[6]))
    for i in 1:4
]

# ─── Figure: PLA ──────────────────────────────────────────────────────────────
fig_pla = let f = Figure()
    positions  = [(1,1), (1,2), (2,1), (2,2)]
    param_names = ["k1", "k5", "k6", "k8"]
    for (i, (pos, param_name)) in enumerate(zip(positions, param_names))
        ax = Axis(getindex(f, pos...), xlabel=param_name, ylabel="ΔL", title="Profile Likelihood for $param_name")
        likelihood_profile!(ax, pla_results[i], i, param_name)
        scatter!(ax, [final[i]], [0], color=Makie.wong_colors()[2])
        hlines!(ax, [quantile(Chisq(1), 0.95)], linestyle=:dash, color=Makie.wong_colors()[3])
        ylims!(ax, (0.0, 5.0))
    end
    f
end
save(joinpath(@__DIR__, "fig_ogtt4_pla.png"), fig_pla)

println("Done. Figures saved: fig_ogtt4_data.png, fig_ogtt4_fit.png, fig_ogtt4_hist.png, fig_ogtt4_pla.png")
