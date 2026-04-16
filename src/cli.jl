# src/cli.jl
# CLI parsing and path resolution

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

function resolve_input_path(input_dir::String, filename::String)
    return isabspath(filename) ? filename : joinpath(input_dir, filename)
end

function resolve_output_path(out_dir::String, filename::String)
    return isabspath(filename) ? filename : joinpath(out_dir, filename)
end
