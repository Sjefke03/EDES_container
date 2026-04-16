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

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

cli = parse_cli(ARGS)

input_dir = get(ENV, "EDES_INPUT_DIR", joinpath(@__DIR__, "inputs"))

# Always load scenario data from input files (defaults to test_data_<scenario>.json)
data_file = cli.data_file === nothing ? default_data_filename(cli.scenario) : cli.data_file
data_path = resolve_input_path(input_dir, data_file)
println("[INFO] Loading data from: $data_path")
custom_data, predefined_params = load_data_json(data_path, cli.scenario)
println("[INFO] Data loaded successfully: $(size(custom_data)) matrix")

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

output_dir = get(ENV, "EDES_OUTPUT_DIR", joinpath(@__DIR__, "outputs"))
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
