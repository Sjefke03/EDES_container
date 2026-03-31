# edes_universal_runner.jl
# Universal EDES runner for: cgm, ogtt3, ogtt4
# Run from repo root: julia runnable/universal/edes_universal_runner.jl [flags]

project_dir = get(ENV, "EDES_PROJECT_DIR", @__DIR__)
import Pkg; Pkg.activate(project_dir)

if get(ENV, "EDES_AUTO_PKG", "1") == "1"
    Pkg.resolve()
    Pkg.instantiate()
end
using OrdinaryDiffEq, CairoMakie, QuasiMonteCarlo
using Optimization, OptimizationOptimJL, JSON
using DelimitedFiles
import OptimizationOptimJL: LBFGS, NelderMead

# -----------------------------------------------------------------------------
# Shared physiology constants
# -----------------------------------------------------------------------------

bw = 70.0
Dmeal = 75.0e3
k2 = 0.28
k3 = 6.07e-3
k4 = 2.35e-4
k7 = 1.15
k9 = 3.83e-2
k10 = 2.84e-1
tau_i = 31.0
tau_d = 3.0
beta = 1.0
Gren = 9.0
EGPb = 0.043
Km = 13.2
f = 0.005551
c1 = 0.1
sigma = 1.4

# -----------------------------------------------------------------------------
# Scenario data
# -----------------------------------------------------------------------------

function build_scenario_data(scenario::String, custom_data::Union{Nothing, Matrix{Float64}} = nothing)
    if scenario == "cgm"
        if custom_data === nothing
            cgm_time = [0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125]
            cgm_glu = [6.1 6.2 6.4 6.7 7.1 7.6 8.1 8.5 8.9 9.2 9.4 9.5 9.6 9.6 9.5 9.4 9.3 9.2 9.0 8.9 8.8 8.6 8.5 8.3 8.2 7.9]
            fasting_ins = 7.9
            cgm_ins = [fasting_ins zeros(1, length(cgm_time) - 1)]
            data = [cgm_time; cgm_glu; cgm_ins]
        else
            data = custom_data
        end

        return Dict(
            "scenario" => "cgm",
            "data" => data,
            "Gb" => data[2, 1],
            "Ib" => data[3, 1],
            "k8_mode" => :fixed,
            "k8_fixed" => 7.25,
            "param_names" => ["k1", "k5", "k6", "sigma_g", "sigma_i"],
            "model_param_count" => 3,
            "lb" => [0.001, 0.001, 0.5, 0.01, 0.1],
            "ub" => [0.1, 0.15, 5.0, 2.0, 50.0],
            "save_idxs" => [2],
            "uses_insulin_data" => false,
            "n_initial_guesses" => 200,
        )
    elseif scenario == "ogtt3"
        if custom_data === nothing
            data_time = [0 15 30 45 60 90 120]
            data_glu = [5.4 7.1 8.6 8.9 8.5 7.6 6.6]
            data_ins = [7.9 36.7 64.8 75.6 79.6 80.5 68.7]
            data = [data_time; data_glu; data_ins]
        else
            data = custom_data
        end

        return Dict(
            "scenario" => "ogtt3",
            "data" => data,
            "Gb" => data[2, 1],
            "Ib" => data[3, 1],
            "k8_mode" => :fixed,
            "k8_fixed" => 7.25,
            "param_names" => ["k1", "k5", "k6", "sigma_g", "sigma_i"],
            "model_param_count" => 3,
            "lb" => [0.001, 0.001, 0.5, 0.01, 0.1],
            "ub" => [0.1, 0.15, 5.0, 2.0, 50.0],
            "save_idxs" => [2, 3],
            "uses_insulin_data" => true,
            "n_initial_guesses" => 300,
        )
    elseif scenario == "ogtt4"
        if custom_data === nothing
            data_time = [0 15 30 45 60 90 120]
            data_glu = [5.4 7.1 8.6 8.9 8.5 7.6 6.6]
            data_ins = [7.9 36.7 64.8 75.6 79.6 80.5 68.7]
            data = [data_time; data_glu; data_ins]
        else
            data = custom_data
        end

        return Dict(
            "scenario" => "ogtt4",
            "data" => data,
            "Gb" => data[2, 1],
            "Ib" => data[3, 1],
            "k8_mode" => :free,
            "k8_fixed" => 7.25,
            "param_names" => ["k1", "k5", "k6", "k8", "sigma_g", "sigma_i"],
            "model_param_count" => 4,
            "lb" => [0.001, 0.001, 0.5, 0.1, 0.01, 0.1],
            "ub" => [0.1, 0.15, 5.0, 25.0, 2.0, 50.0],
            "save_idxs" => [2, 3],
            "uses_insulin_data" => true,
            "n_initial_guesses" => 1000,
        )
    else
        error("Unknown scenario '$scenario'. Use one of: cgm, ogtt3, ogtt4")
    end
end

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

function print_usage()
    println("Usage: julia runnable/universal/edes_universal_runner.jl [flags]")
    println()
    println("Flags:")
    println("  -h, --help                 Show help")
    println("  -scenario NAME             Scenario: cgm | ogtt3 | ogtt4 (default: cgm)")
    println("  -data FILE                 Load real data from JSON file")
    println("  -params CSV                Predefined parameters (scenario-specific)")
    println("  -json [filename]           Save JSON output (default: results_<scenario>.json)")
    println("  -image [filename]          Save figure PNG (default: fig_<scenario>_curves.png)")
    println()
    println("Parameter formats by scenario:")
    println("  cgm   : k1,k5,k6,sigma_g,sigma_i")
    println("  ogtt3 : k1,k5,k6,sigma_g,sigma_i")
    println("  ogtt4 : k1,k5,k6,k8,sigma_g,sigma_i")
    println()
    println("Data file format (JSON):")
    println("  {")
    println("    \"time\": [...],")
    println("    \"glucose\": [...],")
    println("    \"insulin\": [...]")
    println("  }")
    println()
    println("Examples:")
    println("  julia runnable/universal/edes_universal_runner.jl -scenario cgm -json -image")
    println("  julia runnable/universal/edes_universal_runner.jl -scenario ogtt3 -params 0.01,0.05,3.0,0.5,0.3 -json ogtt3.json")
    println("  julia runnable/universal/edes_universal_runner.jl -scenario ogtt4 -params 0.01,0.05,3.0,7.5,0.5,0.3 -image ogtt4.png")
    println("  julia runnable/universal/edes_universal_runner.jl -scenario cgm -data mydata.json -json")
end

function parse_params_csv(text::String)
    raw = split(text, ",")
    vals = Float64[]
    for token in raw
        cleaned = strip(token)
        push!(vals, parse(Float64, cleaned))
    end
    any(!isfinite, vals) && error("-params values must all be finite numbers")
    return vals
end

function load_data_json(filepath::String, scenario_hint::String)
    """Load real data from JSON file.
    
    Expected format:
    {
      "scenario": "cgm",  (optional, can override command line)
      "time": [...],
      "glucose": [...],
      "insulin": [...]    (required for ogtt scenarios, optional for cgm)
    }
    """
    !isfile(filepath) && error("Data file not found: $filepath")
    
    content = JSON.parsefile(filepath)
    
    # Validate required fields
    haskey(content, "time") || error("JSON data must contain 'time' array")
    haskey(content, "glucose") || error("JSON data must contain 'glucose' array")
    
    time_data = vec(content["time"])
    glucose_data = vec(content["glucose"])
    
    # Insulin is optional for CGM, but required for OGTT
    if scenario_hint in ["ogtt3", "ogtt4"]
        haskey(content, "insulin") || error("Scenario '$scenario_hint' requires 'insulin' array in JSON")
        insulin_data = vec(content["insulin"])
    else
        # For CGM, use fasting insulin if not provided
        insulin_data = get(content, "insulin", nothing)
        if insulin_data === nothing
            fasting_ins = get(content, "fasting_insulin", 7.9)
            insulin_data = [fasting_ins; zeros(length(time_data) - 1)]
        else
            insulin_data = vec(insulin_data)
        end
    end
    
    # Validate sizes
    length(time_data) == length(glucose_data) || error("time and glucose arrays must have equal length")
    length(time_data) == length(insulin_data) || error("time and insulin arrays must have equal length")
    length(time_data) >= 3 || error("Must have at least 3 data points")
    
    # Check for valid values
    all(isfinite, time_data) || error("time contains non-finite values")
    all(isfinite, glucose_data) || error("glucose contains non-finite values")
    all(isfinite, insulin_data) || error("insulin contains non-finite values")
    
    # Return as matrix compatible with build_scenario_data format
    return convert(Matrix{Float64}, [time_data'; glucose_data'; insulin_data'])
end

function resolve_output_path(out_dir::String, filename::String)
    return isabspath(filename) ? filename : joinpath(out_dir, filename)
end

function parse_cli(args::Vector{String})
    scenario = "cgm"
    predefined_params = nothing
    data_file = nothing
    emit_json = false
    emit_image = false
    json_filename = ""
    image_filename = ""

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_usage()
            exit(0)
        elseif arg == "-scenario"
            i < length(args) || error("Missing value after -scenario")
            scenario = lowercase(strip(args[i + 1]))
            i += 2
        elseif arg == "-data"
            i < length(args) || error("Missing value after -data")
            data_file = args[i + 1]
            i += 2
        elseif arg == "-params"
            i < length(args) || error("Missing value after -params")
            predefined_params = parse_params_csv(args[i + 1])
            i += 2
        elseif arg == "-json"
            emit_json = true
            if i < length(args) && !startswith(args[i + 1], "-")
                json_filename = args[i + 1]
                i += 2
            else
                i += 1
            end
        elseif arg == "-image"
            emit_image = true
            if i < length(args) && !startswith(args[i + 1], "-")
                image_filename = args[i + 1]
                i += 2
            else
                i += 1
            end
        else
            error("Unknown flag: $arg. Use -h for help.")
        end
    end

    return (;
        scenario = scenario,
        data_file = data_file,
        predefined_params = predefined_params,
        emit_json = emit_json,
        emit_image = emit_image,
        json_filename = json_filename,
        image_filename = image_filename,
    )
end

function validate_params!(params::Vector{Float64}, cfg::Dict)
    expected = length(cfg["param_names"])
    length(params) == expected || error("Scenario $(cfg["scenario"]) expects $expected parameters, got $(length(params)).")

    lb = cfg["lb"]
    ub = cfg["ub"]
    names = cfg["param_names"]
    for i in eachindex(params)
        if params[i] < lb[i] || params[i] > ub[i]
            error("Parameter $(names[i])=$(params[i]) is outside bounds [$(lb[i]), $(ub[i])].")
        end
    end
end

# -----------------------------------------------------------------------------
# Model and loss
# -----------------------------------------------------------------------------

function build_constants(cfg::Dict)
    Vg = 17.0 / bw
    return [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, cfg["Gb"], cfg["Ib"]]
end

function edesode!(du, u, p, t)
    Ggut, Gpl, Ipl, Irem = u
    k1, k2v, k3v, k4v, k5, k6, k7v, k8, k9v, k10v, tau_iv, tau_dv, betav, Grenv, EGPbv, Kmv, fv, Vgv, c1v, sigmav, Dmealv, bwv, Gbv, Ibv = p

    du[1] = sigmav * k1^sigmav * t^(sigmav - 1) * exp(-(k1 * t)^sigmav) * Dmealv - k2v * Ggut

    gliv = EGPbv - k3v * (Gpl - Gbv) - k4v * betav * Irem
    ggut = k2v * (fv / (Vgv * bwv)) * Ggut
    u_ii = EGPbv * ((Kmv + Gbv) / Gbv) * (Gpl / (Kmv + Gpl))
    u_id = k5 * betav * Irem * (Gpl / (Kmv + Gpl))
    u_ren = c1v / (Vgv * bwv) * (Gpl - Grenv) * (Gpl > Grenv)
    du[2] = gliv + ggut - u_ii - u_id - u_ren

    i_pnc = betav^(-1) * (k6 * (Gpl - Gbv) + (k7v / tau_iv) * Gbv + k8 * tau_dv * du[2])
    i_liv = k7v * Gbv * Ipl / (betav * tau_iv * Ibv)
    i_int = k9v * (Ipl - Ibv)
    du[3] = i_pnc - i_liv - i_int
    du[4] = i_int - k10v * Irem
end

function construct_parameters(theta::Vector{Float64}, constants::Vector{Float64}, cfg::Dict)
    if cfg["k8_mode"] == :fixed
        k1v = theta[1]
        k5v = theta[2]
        k6v = theta[3]
        k8v = cfg["k8_fixed"]
    else
        k1v = theta[1]
        k5v = theta[2]
        k6v = theta[3]
        k8v = theta[4]
    end

    k2v = constants[1]; k3v = constants[2]; k4v = constants[3]; k7v = constants[4]
    k9v = constants[5]; k10v = constants[6]; tau_iv = constants[7]; tau_dv = constants[8]
    betav = constants[9]; Grenv = constants[10]; EGPbv = constants[11]; Kmv = constants[12]
    fv = constants[13]; Vgv = constants[14]; c1v = constants[15]; sigmav = constants[16]
    Dmealv = constants[17]; bwv = constants[18]; Gbv = constants[19]; Ibv = constants[20]

    return [k1v, k2v, k3v, k4v, k5v, k6v, k7v, k8v, k9v, k10v, tau_iv, tau_dv, betav, Grenv, EGPbv, Kmv, fv, Vgv, c1v, sigmav, Dmealv, bwv, Gbv, Ibv]
end

ode_solver = Tsit5()

function loss_universal(theta::Vector{Float64}, payload)
    problem, constants, data, cfg = payload

    glucose_data = data[2, :]
    insulin_data = data[3, :]
    data_timepoints = data[1, :]

    # Check parameter bounds and return large penalty if violated
    for (i, val) in enumerate(theta)
        if val < cfg["lb"][i] || val > cfg["ub"][i]
            return 1e10  # Large penalty for out-of-bounds
        end
    end

    p = construct_parameters(theta, constants, cfg)
    local pred
    try
        pred = solve(problem, ode_solver, p = p, saveat = data_timepoints, save_idxs = cfg["save_idxs"],
                     u0 = [0.0, data[2, 1], data[3, 1], 0.0])
    catch
        return 1e10
    end

    if cfg["uses_insulin_data"]
        sol = Array(pred)
        size(sol, 2) != length(glucose_data) && return 1e10

        # Raw residuals (no normalization for likelihood)
        g_resid = sol[1, :] - glucose_data
        i_resid = sol[2, :] - insulin_data
        any(!isfinite, g_resid) && return 1e10
        any(!isfinite, i_resid) && return 1e10

        n = length(glucose_data)
        sigma_g = theta[end - 1]
        sigma_i = theta[end]
        
        # Proper negative log-likelihood for normal distribution
        L_g = n * log(sigma_g) + sum(abs2, g_resid) / (2 * sigma_g^2)
        L_i = n * log(sigma_i) + sum(abs2, i_resid) / (2 * sigma_i^2)
        (isfinite(L_g) && isfinite(L_i)) || return 1e10
        return L_g + L_i
    else
        sol = vec(Array(pred))
        length(sol) == length(glucose_data) || return 1e10

        # Raw residuals (no normalization for likelihood)
        g_resid = sol - glucose_data
        any(!isfinite, g_resid) && return 1e10

        n = length(glucose_data)
        sigma_g = theta[end - 1]
        
        # Proper negative log-likelihood for normal distribution
        L_g = n * log(sigma_g) + sum(abs2, g_resid) / (2 * sigma_g^2)
        isfinite(L_g) || return 1e10
        return L_g
    end
end

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

cli = parse_cli(ARGS)

# Load custom data if provided
custom_data = nothing
if cli.data_file !== nothing
    println("[INFO] Loading data from: $(cli.data_file)")
    custom_data = load_data_json(cli.data_file, cli.scenario)
    println("[INFO] Data loaded successfully: $(size(custom_data)) matrix")
end

cfg = build_scenario_data(cli.scenario, custom_data)
constants = build_constants(cfg)
data = cfg["data"]

# Log measurement data being used
println("[INFO] Measurement data for fitting:")
println("[INFO]   Timepoints: $(data[1, :])")
println("[INFO]   Glucose data ($(size(data, 2)) points): $(data[2, :])")
if cfg["uses_insulin_data"]
    println("[INFO]   Insulin data ($(size(data, 2)) points): $(data[3, :])")
else
    println("[INFO]   Insulin: Fixed at fasting level")
end

if cli.predefined_params !== nothing
    validate_params!(cli.predefined_params, cfg)
end

prob = ODEProblem(edesode!, [0.0, cfg["Gb"], cfg["Ib"], 0.0], (0.0, 240.0), constants)

if cli.predefined_params === nothing
    println("[INFO] Scenario=$(cfg["scenario"]) with no -params. Running optimization with measurement data.")
    if cfg["uses_insulin_data"]
        println("[INFO] Fitting to both glucose and insulin measurements")
    else
        println("[INFO] Fitting to glucose measurements only")
    end

    lhs = LatinHypercubeSample()
    initial_guess = QuasiMonteCarlo.sample(cfg["n_initial_guesses"], cfg["lb"], cfg["ub"], lhs)

    optf = OptimizationFunction(loss_universal)
    results = Any[]
    local failed_count = 0

    for (idx, guess) in enumerate(eachcol(initial_guess))
        if idx == 1 || idx % 25 == 0
            println("[INFO] Optimization run $idx/$(cfg["n_initial_guesses"])")
        end
        try
            res = solve(OptimizationProblem(optf, Vector(guess), (prob, constants, data, cfg)), NelderMead())
            if res.objective > 0 && isfinite(res.objective)
                push!(results, res)
            else
                failed_count += 1
            end
        catch
            failed_count += 1
        end
    end

    println("[INFO] Optimization complete. Successful runs=$(length(results))/$(cfg["n_initial_guesses"])")
    isempty(results) && error("All optimization runs failed for scenario $(cfg["scenario"]).")
    
    best_idx = argmin([r.objective for r in results])
    final_params = results[best_idx].u
    println("[OK] Best objective=$(results[best_idx].objective)")
else
    final_params = cli.predefined_params
    println("[OK] Scenario=$(cfg["scenario"]) using validated predefined parameters.")
end

name_value = String[]
for (name, value) in zip(cfg["param_names"], final_params)
    push!(name_value, "$(name)=$(value)")
end
println("[RESULT] " * join(name_value, " "))

solution = solve(prob, ode_solver, p = construct_parameters(final_params, constants, cfg),
                 u0 = [0.0, data[2, 1], data[3, 1], 0.0])
println("[OK] Simulation complete for $(length(solution.t)) time points")

output_dir = @__DIR__
default_json_name = "results_$(cfg["scenario"]).json"
default_image_name = "fig_$(cfg["scenario"])_curves.png"

if cli.emit_json
    json_name = isempty(cli.json_filename) ? default_json_name : cli.json_filename
    json_path = resolve_output_path(output_dir, json_name)

    param_dict = Dict{String, Float64}()
    for (name, value) in zip(cfg["param_names"], final_params)
        param_dict[name] = value
    end

    payload = Dict(
        "scenario" => cfg["scenario"],
        "parameter_source" => (cli.predefined_params === nothing ? "optimized" : "predefined"),
        "parameters" => param_dict,
        "simulation" => Dict(
            "time" => collect(solution.t),
            "gut_glucose" => collect(solution[1, :]),
            "plasma_glucose" => collect(solution[2, :]),
            "plasma_insulin" => collect(solution[3, :]),
            "interstitium_insulin" => collect(solution[4, :]),
        ),
        "data" => Dict(
            "time" => vec(data[1, :]),
            "glucose" => vec(data[2, :]),
            "insulin" => vec(data[3, :]),
        ),
    )

    open(json_path, "w") do fjson
        write(fjson, JSON.json(payload, 2))
    end
    println("[OK] JSON saved: $json_path")
else
    println("[INFO] JSON output disabled. Use -json to enable it.")
end

if cli.emit_image
    image_name = isempty(cli.image_filename) ? default_image_name : cli.image_filename
    image_path = resolve_output_path(output_dir, image_name)

    fig = Figure(size = (500, 500))
    ax_g_gut = Axis(fig[1, 1], xlabel = "Time [min]", ylabel = "Glucose Mass [mg/dL]", title = "Gut Glucose")
    ax_g_plasma = Axis(fig[1, 2], xlabel = "Time [min]", ylabel = "Glucose Concentration [mM]", title = "Plasma Glucose")
    ax_i_plasma = Axis(fig[2, 1], xlabel = "Time [min]", ylabel = "Insulin Concentration [mU/L]", title = "Plasma Insulin")
    ax_i_int = Axis(fig[2, 2], xlabel = "Time [min]", ylabel = "Insulin Concentration [mU/L]", title = "Interstitium Insulin")

    lines!(ax_g_gut, solution.t, solution[1, :], linewidth = 2, color = :navy)
    lines!(ax_g_plasma, solution.t, solution[2, :], linewidth = 2, color = :navy)
    lines!(ax_i_plasma, solution.t, solution[3, :], linewidth = 2, color = :navy)
    lines!(ax_i_int, solution.t, solution[4, :], linewidth = 2, color = :navy)

    sigma_g = final_params[end - 1]
    sigma_i = final_params[end]
    band!(ax_g_plasma, solution.t, solution[2, :] .+ sigma_g, solution[2, :] .- sigma_g, alpha = 0.2, color = :navy)
    band!(ax_i_plasma, solution.t, solution[3, :] .+ sigma_i, solution[3, :] .- sigma_i, alpha = 0.2, color = :navy)

    scatter!(ax_g_plasma, data[1, :], data[2, :], color = :red, markersize = 5, label = "Data")
    if cfg["uses_insulin_data"]
        scatter!(ax_i_plasma, data[1, :], data[3, :], color = :red, markersize = 5)
    end
    axislegend(ax_g_plasma)

    save(image_path, fig)
    println("[OK] Image saved: $image_path")
else
    println("[INFO] Image output disabled. Use -image to enable it.")
end
