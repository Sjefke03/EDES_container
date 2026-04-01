# edes_universal_runner.jl
# Universal EDES runner for: cgm, ogtt3, ogtt4
# Run from this folder: julia edes_universal_runner.jl [flags]

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
    println("Usage: julia edes_universal_runner.jl [flags]")
    println()
    println("Flags:")
    println("  -h, --help                 Show help")
    println("  -scenario NAME             Scenario: cgm | ogtt3 | ogtt4 (default: cgm)")
    println("  -data FILE                 Load data JSON from input directory")
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
    println("    \"insulin\": [...],")
    println("    \"parameters\": [...]   # optional; if present, optimization is skipped")
    println("  }")
    println()
    println("Examples:")
    println("  julia edes_universal_runner.jl -scenario cgm -json -image")
    println("  julia edes_universal_runner.jl -scenario ogtt3 -data test_data_ogtt3.json -json ogtt3.json")
    println("  julia edes_universal_runner.jl -scenario ogtt3 -data test_data_ogtt3_with_params.json -json ogtt3_pretrained.json")
end

function expected_param_names(scenario::String)
    if scenario == "ogtt4"
        return ["k1", "k5", "k6", "k8", "sigma_g", "sigma_i"]
    elseif scenario == "ogtt3" || scenario == "cgm"
        return ["k1", "k5", "k6", "sigma_g", "sigma_i"]
    end
    error("Unknown scenario '$scenario'.")
end

function resolve_input_path(input_dir::String, filename::String)
    return isabspath(filename) ? filename : joinpath(input_dir, filename)
end

function parse_parameters_json(raw_params, scenario::String)
    names = expected_param_names(scenario)

    if raw_params isa AbstractVector
        vals = Float64.(raw_params)
    elseif raw_params isa AbstractDict
        vals = Float64[]
        for n in names
            haskey(raw_params, n) || error("parameters JSON is missing key '$n'")
            push!(vals, Float64(raw_params[n]))
        end
    else
        error("'parameters' must be either a JSON array or object")
    end

    length(vals) == length(names) || error("Scenario '$scenario' expects $(length(names)) parameters, got $(length(vals)).")
    any(!isfinite, vals) && error("parameters contains non-finite values")
    return vals
end

function default_data_filename(scenario::String)
    if scenario == "cgm"
        return "test_data_cgm.json"
    elseif scenario == "ogtt3"
        return "test_data_ogtt3.json"
    elseif scenario == "ogtt4"
        return "test_data_ogtt4.json"
    end
    error("Unknown scenario '$scenario'.")
end

function load_data_json(filepath::String, scenario_hint::String)
    """Load real data from JSON file.

    Supports two formats:

    New ontology format (v2):
    {
      "HDT-EDES-SCENARIO": "cgm",
      "14749-6": { "timestamps_min": [...], "values": [...] },
      "20448-7": { "value": 7.9 },
      "HDT-EDES-INSULIN": { "timestamps_min": [...], "values": [...] },  // ogtt3/ogtt4
      "HDT-EDES-PARAMS": { "k1": ..., ... }  // optional
    }

    Legacy format:
    {
      "time": [...],
      "glucose": [...],
      "insulin": [...],    // required for ogtt scenarios
      "fasting_insulin": 7.9,
      "parameters": [...]  // optional
    }
    """
    !isfile(filepath) && error("Data file not found: $filepath")

    content = JSON.parsefile(filepath)

    if haskey(content, "HDT-EDES-SCENARIO")
        # ── New ontology format ──────────────────────────────────────────────
        scenario_hint = content["HDT-EDES-SCENARIO"]

        haskey(content, "14749-6") || error("New-format JSON must contain '14749-6' (glucose time series)")
        glucose_block = content["14749-6"]
        time_data    = vec(Float64.(glucose_block["timestamps_min"]))
        glucose_data = vec(Float64.(glucose_block["values"]))

        if scenario_hint in ["ogtt3", "ogtt4"]
            haskey(content, "HDT-EDES-INSULIN") ||
                error("Scenario '$scenario_hint' requires 'HDT-EDES-INSULIN' in new-format JSON")
            insulin_data = vec(Float64.(content["HDT-EDES-INSULIN"]["values"]))
        else
            insulin_block = get(content, "20448-7", Dict())
            fasting_ins   = get(insulin_block, "value", 7.9)
            insulin_data  = [Float64(fasting_ins); zeros(length(time_data) - 1)]
        end

        params_from_json = haskey(content, "HDT-EDES-PARAMS") ?
            parse_parameters_json(content["HDT-EDES-PARAMS"], scenario_hint) : nothing

        # Meal dose: HDT-EDES-MEAL-DOSE.value in mg (default 75 000 mg = 75 g)
        meal_dose_mg = Float64(get(get(content, "HDT-EDES-MEAL-DOSE", Dict()), "value", Dmeal))

    else
        # ── Legacy format ────────────────────────────────────────────────────
        haskey(content, "time")    || error("JSON data must contain 'time' array")
        haskey(content, "glucose") || error("JSON data must contain 'glucose' array")

        time_data    = vec(Float64.(content["time"]))
        glucose_data = vec(Float64.(content["glucose"]))

        if scenario_hint in ["ogtt3", "ogtt4"]
            haskey(content, "insulin") ||
                error("Scenario '$scenario_hint' requires 'insulin' array in JSON")
            insulin_data = vec(Float64.(content["insulin"]))
        else
            raw_ins = get(content, "insulin", nothing)
            if raw_ins === nothing
                fasting_ins  = get(content, "fasting_insulin", 7.9)
                insulin_data = [Float64(fasting_ins); zeros(length(time_data) - 1)]
            else
                insulin_data = vec(Float64.(raw_ins))
            end
        end

        params_from_json = nothing
        if haskey(content, "parameters")
            params_from_json = parse_parameters_json(content["parameters"], scenario_hint)
        end
        meal_dose_mg = Dmeal   # legacy format has no meal dose field
    end

    # Common validation
    length(time_data) == length(glucose_data) || error("time and glucose arrays must have equal length")
    length(time_data) == length(insulin_data) || error("time and insulin arrays must have equal length")
    length(time_data) >= 3 || error("Must have at least 3 data points")
    all(isfinite, time_data)    || error("time contains non-finite values")
    all(isfinite, glucose_data) || error("glucose contains non-finite values")
    all(isfinite, insulin_data) || error("insulin contains non-finite values")

    return convert(Matrix{Float64}, [time_data'; glucose_data'; insulin_data']), params_from_json, scenario_hint, meal_dose_mg
end

function resolve_output_path(out_dir::String, filename::String)
    return isabspath(filename) ? filename : joinpath(out_dir, filename)
end

function parse_cli(args::Vector{String})
    scenario = "cgm"
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

function build_constants(cfg::Dict, meal_dose::Float64 = Dmeal)
    Vg = 17.0 / bw
    return [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, meal_dose, bw, cfg["Gb"], cfg["Ib"]]
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

input_dir = get(ENV, "EDES_INPUT_DIR", joinpath(@__DIR__, "inputs"))

# Check for input.json (docker runner mode) — overrides CLI data_file and scenario
input_json_candidate = joinpath(input_dir, "input.json")
if isfile(input_json_candidate) && cli.data_file === nothing
    data_path = input_json_candidate
else
    data_file = cli.data_file === nothing ? default_data_filename(cli.scenario) : cli.data_file
    data_path = resolve_input_path(input_dir, data_file)
end

println("[INFO] Loading data from: $data_path")
custom_data, predefined_params, effective_scenario, meal_dose_mg = load_data_json(data_path, cli.scenario)

# Use scenario from file if it differs from the CLI default
if effective_scenario != cli.scenario
    println("[INFO] Scenario overridden by input file: $(cli.scenario) -> $effective_scenario")
    cli = (scenario=effective_scenario, data_file=cli.data_file, emit_json=cli.emit_json,
           emit_image=cli.emit_image, json_filename=cli.json_filename, image_filename=cli.image_filename)
end

println("[INFO] Data loaded successfully: $(size(custom_data)) matrix")

cfg = build_scenario_data(cli.scenario, custom_data)
println("[INFO] Meal dose: $(meal_dose_mg / 1000.0) g ($(meal_dose_mg) mg)")
constants = build_constants(cfg, meal_dose_mg)
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

if predefined_params !== nothing
    validate_params!(predefined_params, cfg)
    println("[INFO] Pretrained parameters found in JSON. Optimization will be skipped.")
else
    println("[INFO] No pretrained parameters in JSON. Optimization will run.")
end

prob = ODEProblem(edesode!, [0.0, cfg["Gb"], cfg["Ib"], 0.0], (0.0, 240.0), constants)

if predefined_params === nothing
    println("[INFO] Scenario=$(cfg["scenario"]) with no pretrained parameters. Running optimization with measurement data.")
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
    final_params = predefined_params
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

output_dir = get(ENV, "EDES_OUTPUT_DIR", joinpath(@__DIR__, "outputs"))
default_json_name = "results_$(cfg["scenario"]).json"
default_image_name = "fig_$(cfg["scenario"])_curves.png"

# Build parameter dict for outputs
param_dict = Dict{String, Float64}()
for (name, value) in zip(cfg["param_names"], final_params)
    param_dict[name] = value
end

# Compute diagnoses
fasting_glucose = data[2, 1]
peak_glucose    = maximum(solution[2, :])

diagnoses = Dict{String, Any}[]
if peak_glucose >= 11.1
    push!(diagnoses, Dict("ontology_term_code" => "44054006",
                          "name" => "Type 2 Diabetes Mellitus Risk",
                          "present" => true,
                          "evidence" => "Peak simulated plasma glucose $(round(peak_glucose, digits=2)) mmol/L (>= 11.1)"))
elseif peak_glucose >= 7.8
    push!(diagnoses, Dict("ontology_term_code" => "9414007",
                          "name" => "Impaired Glucose Tolerance",
                          "present" => true,
                          "evidence" => "Peak simulated plasma glucose $(round(peak_glucose, digits=2)) mmol/L (7.8–11.1 range)"))
else
    push!(diagnoses, Dict("ontology_term_code" => "HDT-DIAG-NORMAL-GLUCOSE",
                          "name" => "Normal Glucose Regulation",
                          "present" => true,
                          "evidence" => "Peak $(round(peak_glucose, digits=2)) mmol/L < 7.8 and fasting $(round(fasting_glucose, digits=2)) mmol/L <= 6.1"))
end
if fasting_glucose > 6.1
    push!(diagnoses, Dict("ontology_term_code" => "HDT-DIAG-ELEVATED-FASTING-GLUCOSE",
                          "name" => "Elevated Fasting Glucose",
                          "present" => true,
                          "evidence" => "Fasting glucose $(round(fasting_glucose, digits=2)) mmol/L > 6.1"))
end

# Compute advices
advices = Dict{String, Any}[]
if peak_glucose >= 11.1
    push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-CONSULT-HCP",
                        "name" => "Consult Healthcare Provider",
                        "message" => "Consult a healthcare provider for further clinical evaluation and management."))
    push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-REDUCE-CARBS",
                        "name" => "Reduce Carbohydrate Intake",
                        "message" => "Reduce dietary carbohydrate intake to lower post-meal glucose excursions."))
elseif peak_glucose >= 7.8
    push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-INCREASE-ACTIVITY",
                        "name" => "Increase Physical Activity",
                        "message" => "Increase aerobic physical activity to improve insulin sensitivity and glucose tolerance."))
    push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-MONITOR-GLUCOSE",
                        "name" => "Increase Monitoring Frequency",
                        "message" => "Increase the frequency of continuous glucose monitoring."))
end

# Always write ontology-coded output.json to EDES_OUTPUT_DIR
ontology_payload = Dict(
    "outputs" => Dict(
        "HDT-EDES-PLASMA-GLUCOSE" => Dict(
            "timestamps_min" => collect(solution.t),
            "values"         => collect(solution[2, :]),
            "unit"           => "mmol/L",
        ),
        "HDT-EDES-PLASMA-INSULIN" => Dict(
            "timestamps_min" => collect(solution.t),
            "values"         => collect(solution[3, :]),
            "unit"           => "mU/L",
        ),
        "HDT-EDES-GUT-GLUCOSE" => Dict(
            "timestamps_min" => collect(solution.t),
            "values"         => collect(solution[1, :]),
            "unit"           => "mg",
        ),
        "HDT-EDES-INTERSTITIUM-INSULIN" => Dict(
            "timestamps_min" => collect(solution.t),
            "values"         => collect(solution[4, :]),
            "unit"           => "mU/L",
        ),
        "HDT-EDES-FIT-PARAMS" => param_dict,
    ),
    "diagnoses" => diagnoses,
    "advices"   => advices,
)

mkpath(output_dir)
output_json_path = joinpath(output_dir, "output.json")
open(output_json_path, "w") do f
    write(f, JSON.json(ontology_payload, 2))
end
println("[OK] output.json saved: $output_json_path")

if cli.emit_json
    json_name = isempty(cli.json_filename) ? default_json_name : cli.json_filename
    json_path = resolve_output_path(output_dir, json_name)

    legacy_payload = Dict(
        "scenario" => cfg["scenario"],
        "parameter_source" => (predefined_params === nothing ? "optimized" : "predefined"),
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
        write(fjson, JSON.json(legacy_payload, 2))
    end
    println("[OK] Legacy JSON saved: $json_path")
else
    println("[INFO] Legacy JSON output disabled. Use -json to enable it.")
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
