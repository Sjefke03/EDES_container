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

    # Support two input formats:
    #   HDT ontology format  – keys are ontology term codes (platform sends this)
    #   Native EDES format   – keys are "time" / "glucose" / "insulin" (local dev)
    if haskey(content, "14749-6")
        # --- HDT ontology format ---
        glucose_ts   = content["14749-6"]
        time_data    = vec(glucose_ts["timestamps_min"])
        glucose_data = vec(glucose_ts["values"])

        haskey(content, "20448-7") ||
            error("Scenario '$scenario_hint' requires '20448-7' (insulin) in input")
        ins_block = content["20448-7"]
        if haskey(ins_block, "timestamps_min")
            # Time series format (ogtt3/ogtt4)
            insulin_data = vec(ins_block["values"])
        else
            # Scalar fasting insulin format (cgm)
            if scenario_hint in ["ogtt3", "ogtt4"]
                error("Scenario '$scenario_hint' requires '20448-7' to be a time series (with timestamps_min and values)")
            end
            fasting_ins = Float64(ins_block["value"])
            insulin_data = [fasting_ins; zeros(length(time_data) - 1)]
        end

        params_from_json = nothing
        if haskey(content, "HDT-EDES-PARAMS")
            params_from_json = parse_parameters_json(content["HDT-EDES-PARAMS"], scenario_hint)
        end
    else
        # --- Native EDES format ---
        haskey(content, "time")    || error("JSON data must contain 'time' array")
        haskey(content, "glucose") || error("JSON data must contain 'glucose' array")

        time_data    = vec(content["time"])
        glucose_data = vec(content["glucose"])

        if scenario_hint in ["ogtt3", "ogtt4"]
            haskey(content, "insulin") ||
                error("Scenario '$scenario_hint' requires 'insulin' array in JSON")
            insulin_data = vec(content["insulin"])
        else
            insulin_data = get(content, "insulin", nothing)
            if insulin_data === nothing
                fasting_ins = get(content, "fasting_insulin", 7.9)
                insulin_data = [fasting_ins; zeros(length(time_data) - 1)]
            else
                insulin_data = vec(insulin_data)
            end
        end

        params_from_json = nothing
        if haskey(content, "parameters")
            params_from_json = parse_parameters_json(content["parameters"], scenario_hint)
        end
    end

    # Validate sizes and values
    length(time_data) == length(glucose_data) || error("time and glucose arrays must have equal length")
    length(time_data) == length(insulin_data) || error("time and insulin arrays must have equal length")
    length(time_data) >= 3 || error("Must have at least 3 data points")
    all(isfinite, time_data)    || error("time contains non-finite values")
    all(isfinite, glucose_data) || error("glucose contains non-finite values")
    all(isfinite, insulin_data) || error("insulin contains non-finite values")

    return convert(Matrix{Float64}, [time_data'; glucose_data'; insulin_data']), params_from_json
end

function load_ontology(path::String)
    isfile(path) || error("ontology.json not found at: $path")
    return JSON.parsefile(path)
end
