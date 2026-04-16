# src/scenario.jl
# Scenario configuration: build_scenario_data, expected_param_names, default_data_filename

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
            "n_initial_guesses" => 300,
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
            "n_initial_guesses" => 300,
        )
    else
        error("Unknown scenario '$scenario'. Use one of: cgm, ogtt3, ogtt4")
    end
end

function expected_param_names(scenario::String)
    if scenario == "ogtt4"
        return ["k1", "k5", "k6", "k8", "sigma_g", "sigma_i"]
    elseif scenario == "ogtt3" || scenario == "cgm"
        return ["k1", "k5", "k6", "sigma_g", "sigma_i"]
    end
    error("Unknown scenario '$scenario'.")
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
