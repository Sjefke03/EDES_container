# src/data_io.jl
# Data loading and parameter parsing from JSON

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

function load_data_json(filepath::String, scenario_hint::String)
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

    params_from_json = nothing
    if haskey(content, "parameters")
        params_from_json = parse_parameters_json(content["parameters"], scenario_hint)
    end

    # Return matrix compatible with build_scenario_data format and optional parameters
    return convert(Matrix{Float64}, [time_data'; glucose_data'; insulin_data']), params_from_json
end
