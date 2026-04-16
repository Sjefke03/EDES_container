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

include(joinpath(@__DIR__, "src", "constants.jl"))
include(joinpath(@__DIR__, "src", "scenario.jl"))
include(joinpath(@__DIR__, "src", "model.jl"))
include(joinpath(@__DIR__, "src", "loss.jl"))
include(joinpath(@__DIR__, "src", "cli.jl"))
include(joinpath(@__DIR__, "src", "data_io.jl"))
include(joinpath(@__DIR__, "src", "metrics.jl"))
include(joinpath(@__DIR__, "src", "diagnoses.jl"))

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

# When running via the platform the scenario is carried inside the input JSON
# (HDT-EDES-SCENARIO field).  Use that value if present; fall back to CLI flag.
scenario = cli.scenario
if isfile(data_path)
    try
        _peek = JSON.parsefile(data_path)
        if haskey(_peek, "HDT-EDES-SCENARIO")
            global scenario = lowercase(strip(string(_peek["HDT-EDES-SCENARIO"])))
            println("[INFO] Scenario overridden from input payload: $scenario")
        else
            println("[WARN] HDT-EDES-SCENARIO not found in input JSON — using CLI default: $scenario")
        end
    catch e
        println("[WARN] Could not peek scenario from $data_path: $e — using CLI default: $scenario")
    end
end

custom_data, predefined_params = load_data_json(data_path, scenario)
println("[INFO] Data loaded successfully: $(size(custom_data)) matrix")

cfg = build_scenario_data(scenario, custom_data)
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
            if isfinite(res.objective)
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

ontology = load_ontology(joinpath(@__DIR__, "ontology.json"))
metrics  = compute_metrics(solution, data, cfg, final_params)
diagnoses = compute_diagnoses(metrics, cfg)
println("[OK] Metrics and diagnoses computed")

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

    param_dict = Dict{String, Float64}()
    for (name, value) in zip(cfg["param_names"], final_params)
        param_dict[name] = value
    end

    t_vec = collect(solution.t)
    payload = Dict(
        # ── Platform-standard outputs block (keyed by ontology term codes) ──
        "outputs" => Dict(
            "HDT-EDES-PLASMA-GLUCOSE" => Dict(
                "timestamps_min" => t_vec,
                "values"         => collect(solution[2, :]),
                "unit"           => "mmol/L",
            ),
            "HDT-EDES-PLASMA-INSULIN" => Dict(
                "timestamps_min" => t_vec,
                "values"         => collect(solution[3, :]),
                "unit"           => "mU/L",
            ),
            "HDT-EDES-GUT-GLUCOSE" => Dict(
                "timestamps_min" => t_vec,
                "values"         => collect(solution[1, :]),
                "unit"           => "mg",
            ),
            "HDT-EDES-INTERSTITIUM-INSULIN" => Dict(
                "timestamps_min" => t_vec,
                "values"         => collect(solution[4, :]),
                "unit"           => "mU/L",
            ),
            "HDT-EDES-FIT-PARAMS" => param_dict,
        ),
        # ── Clinical outputs ────────────────────────────────────────────────
        "diagnoses" => diagnoses,
        "advices"   => String[],
        # ── Rich extras (kept for debugging / UI charts) ────────────────────
        "metrics"          => metrics,
        "scenario"         => cfg["scenario"],
        "parameter_source" => (predefined_params === nothing ? "optimized" : "predefined"),
        "ontology_ref" => Dict(
            "version" => ontology["version_tag"],
            "file"    => "ontology.json",
            "model"   => ontology["ontology_id"],
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
