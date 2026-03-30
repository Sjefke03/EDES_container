# edes_cgm_runner.jl
# Standalone runner: simulate EDES model with either predefined or optimized parameters
# Run from repo root: julia runnable/edes_cgm/edes_cgm_runner.jl [flags]

import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..")); Pkg.resolve(); Pkg.instantiate()
using OrdinaryDiffEq, CairoMakie, QuasiMonteCarlo
using Optimization, OptimizationOptimJL, JSON
import OptimizationOptimJL: LBFGS

# ═══════════════════════════════════════════════════════════════════════════════
# DATA AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# User inputs
CGM_time    = [0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125]
CGM_glu     = [6.1 6.2 6.4 6.7 7.1 7.6 8.1 8.5 8.9 9.2 9.4 9.5 9.6 9.6 9.5 9.4 9.3 9.2 9.0 8.9 8.8 8.6 8.5 8.3 8.2 7.9]
fasting_ins = 7.9
k8_fixed    = 7.25

CGM_ins  = [fasting_ins zeros(1, length(CGM_time) - 1)]
CGM_data = [CGM_time; CGM_glu; CGM_ins]
Gb = CGM_glu[1]
Ib = fasting_ins

# Fixed physiological parameters
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

# ODE setup
u0_ggut = 0.0
u0_irem = 0.0
u0 = [u0_ggut, Gb, Ib, u0_irem]
tspan = (0.0, 240.0)

# Optimization bounds
k1_lb, k1_ub = 1e-6, 1.0
k5_lb, k5_ub = 1e-6, 0.5
k6_lb, k6_ub = 1e-6, 10.0
sigma_g_lb, sigma_g_ub = 1e-4, 100.0
sigma_i_lb, sigma_i_ub = 1e-4, 100.0
lb = [k1_lb, k5_lb, k6_lb, sigma_g_lb, sigma_i_lb]
ub = [k1_ub, k5_ub, k6_ub, sigma_g_ub, sigma_i_ub]
param_names = ["k1", "k5", "k6", "sigma_g", "sigma_i"]

# Optimization settings
n_initial_guesses = 100
lhs_sampling = LatinHypercubeSample()
penalty_l2_weight = 1e4
penalty_offset = 1e6
loss_scaling = 0.5

# Output settings
fig_width = 500
fig_height = 500
output_dir = @__DIR__

# ═══════════════════════════════════════════════════════════════════════════════
# CLI PARSING
# ═══════════════════════════════════════════════════════════════════════════════

function print_usage()
    println("Usage: julia runnable/edes_cgm/edes_cgm_runner.jl [flags]")
    println()
    println("Flags:")
    println("  -h, --help                Show this help message")
    println("  -params a,b,c,d,e         Use predefined params [k1,k5,k6,sigma_g,sigma_i]")
    println("  -image                    Save curves image as fig_cgm_curves.png")
    println("  -json [filename]          Save JSON output (default filename: results.json)")
    println()
    println("Examples:")
    println("  julia runnable/edes_cgm/edes_cgm_runner.jl")
    println("  julia runnable/edes_cgm/edes_cgm_runner.jl -params 0.01,0.05,3.0,0.5,0.3")
    println("  julia runnable/edes_cgm/edes_cgm_runner.jl -json")
    println("  julia runnable/edes_cgm/edes_cgm_runner.jl -json my_run.json -image")
end

function parse_params_csv(csv_text::String)
    raw = split(csv_text, ",")
    length(raw) == 5 || error("-params must contain exactly 5 comma-separated values.")
    vals = Float64[]
    for token in raw
        cleaned = strip(token)
        push!(vals, parse(Float64, cleaned))
    end
    any(!isfinite, vals) && error("-params values must be finite numbers.")
    return vals
end

function validate_params!(vals::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64}, names::Vector{String})
    for i in eachindex(vals)
        if vals[i] < lower[i] || vals[i] > upper[i]
            error("Parameter $(names[i])=$(vals[i]) is outside bounds [$(lower[i]), $(upper[i])].")
        end
    end
    return nothing
end

function resolve_output_path(name::String)
    return isabspath(name) ? name : joinpath(output_dir, name)
end

function parse_cli(args::Vector{String})
    predefined_params = nothing
    emit_image = false
    emit_json = false
    json_filename = "results.json"

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_usage()
            exit(0)
        elseif arg == "-params"
            i < length(args) || error("Missing value after -params")
            predefined_params = parse_params_csv(args[i + 1])
            i += 2
        elseif arg == "-image"
            emit_image = true
            i += 1
        elseif arg == "-json"
            emit_json = true
            if i < length(args) && !startswith(args[i + 1], "-")
                json_filename = args[i + 1]
                i += 2
            else
                i += 1
            end
        else
            error("Unknown flag: $arg. Use -h for help.")
        end
    end

    return (;
        predefined_params = predefined_params,
        emit_image = emit_image,
        emit_json = emit_json,
        json_filename = json_filename,
    )
end

cli = parse_cli(ARGS)
predefined_params = cli.predefined_params
emit_image = cli.emit_image
emit_json = cli.emit_json
json_filename = cli.json_filename

if predefined_params !== nothing
    validate_params!(predefined_params, lb, ub, param_names)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ODE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

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

function construct_parameters(θ, c)
    k1 = θ[1]
    k5 = θ[2]
    k6 = θ[3]
    k8 = k8_fixed

    k2 = c[1]; k3 = c[2]; k4 = c[3]; k7 = c[4]; k9 = c[5]; k10 = c[6]
    tau_i = c[7]; tau_d = c[8]; beta = c[9]; Gren = c[10]; EGPb = c[11]
    Km = c[12]; f = c[13]; Vg = c[14]; c1 = c[15]; sigma = c[16]
    Dmeal = c[17]; bw = c[18]; Gb = c[19]; Ib = c[20]

    return [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

function loss_CGM(θ, (problem, constants, data))
    glucose_data    = data[2,:]
    data_timepoints = data[1,:]
    penalty = sum(abs2, θ[1:3]) * penalty_l2_weight + penalty_offset

    p = construct_parameters(θ, constants)
    local pred
    try
        pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2],
                     u0=[u0_ggut, data[2,1], data[3,1], u0_irem])
    catch
        return penalty
    end

    sol = vec(Array(pred))
    if any(isnan, sol)
        return penalty
    end

    length(sol) == length(glucose_data) || return penalty
    g_loss = (sol - glucose_data) / maximum(glucose_data)
    any(!isfinite, g_loss) && return penalty

    n   = length(glucose_data)
    sigma_g = θ[4]
    L_g = n*log(sigma_g * sqrt(2π)) + 1/(2sigma_g^2) * sum(abs2, g_loss)
    isfinite(L_g) || return penalty
    return loss_scaling * L_g
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SOURCE: OPTIMIZATION OR PREDEFINED
# ═══════════════════════════════════════════════════════════════════════════════

prob = ODEProblem(edesode!, u0, tspan, constants)

if predefined_params === nothing
    println("[INFO] No -params provided. Running optimization on raw data.")
    initial_guess = QuasiMonteCarlo.sample(n_initial_guesses, lb, ub, lhs_sampling)
    optf = OptimizationFunction(loss_CGM, AutoForwardDiff())
    results = Any[]

    for (idx, guess) in enumerate(eachcol(initial_guess))
        if idx == 1 || idx % 20 == 0
            println("[INFO] Optimization run $idx/$n_initial_guesses")
        end
        try
            res = solve(OptimizationProblem(optf, Vector(guess), (prob, constants, CGM_data), lb=lb, ub=ub), LBFGS())
            push!(results, res)
        catch
            continue
        end
    end

    isempty(results) && error("All optimization runs failed.")
    best_index = argmin([r.objective for r in results])
    final_params = results[best_index].u
    best_loss = results[best_index].objective
    println("[OK] Optimization complete. Best objective: $best_loss")
else
    final_params = predefined_params
    println("[OK] Using validated predefined parameters from -params")
end

println("[RESULT] k1=$(final_params[1]) k5=$(final_params[2]) k6=$(final_params[3]) sigma_g=$(final_params[4]) sigma_i=$(final_params[5])")

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATE
# ═══════════════════════════════════════════════════════════════════════════════

solution = solve(prob, Tsit5(), p=construct_parameters(final_params, constants), u0=[u0_ggut, CGM_data[2,1], fasting_ins, u0_irem])
println("[OK] Simulation completed for $(length(solution.t)) time points")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

if emit_json
    json_path = resolve_output_path(json_filename)
    results_dict = Dict(
        "parameters" => Dict(
            "k1" => final_params[1],
            "k5" => final_params[2],
            "k6" => final_params[3],
            "sigma_g" => final_params[4],
            "sigma_i" => final_params[5]
        ),
        "simulation" => Dict(
            "time" => collect(solution.t),
            "gut_glucose" => collect(solution[1,:]),
            "plasma_glucose" => collect(solution[2,:]),
            "plasma_insulin" => collect(solution[3,:]),
            "interstitium_insulin" => collect(solution[4,:])
        ),
        "data" => Dict(
            "time" => vec(CGM_data[1,:]),
            "glucose" => vec(CGM_data[2,:]),
            "insulin" => vec(CGM_data[3,:])
        )
    )

    open(json_path, "w") do f
        write(f, JSON.json(results_dict, 2))
    end
    println("[OK] JSON saved: $json_path")
else
    println("[INFO] JSON output disabled. Use -json to enable it.")
end

if emit_image
    fig = Figure(size=(fig_width, fig_height))
    ax_g_gut    = Axis(fig[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]",        title="Gut Glucose")
    ax_g_plasma = Axis(fig[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]",   title="Plasma Glucose")
    ax_i_plasma = Axis(fig[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
    ax_i_int    = Axis(fig[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

    lines!(ax_g_gut, solution.t, solution[1,:], linewidth=2, color=:navy)
    lines!(ax_g_plasma, solution.t, solution[2,:], linewidth=2, color=:navy)
    lines!(ax_i_plasma, solution.t, solution[3,:], linewidth=2, color=:navy)
    lines!(ax_i_int, solution.t, solution[4,:], linewidth=2, color=:navy)

    band!(ax_g_plasma, solution.t, solution[2,:] .+ final_params[4], solution[2,:] .- final_params[4], alpha=0.2, color=:navy)
    band!(ax_i_plasma, solution.t, solution[3,:] .+ final_params[5], solution[3,:] .- final_params[5], alpha=0.2, color=:navy)
    scatter!(ax_g_plasma, CGM_data[1,:], CGM_data[2,:], color=:red, markersize=5, label="Data")
    axislegend(ax_g_plasma)

    fig_path = joinpath(output_dir, "fig_cgm_curves.png")
    save(fig_path, fig)
    println("[OK] Image saved: $fig_path")
else
    println("[INFO] Image output disabled. Use -image to enable it.")
end
