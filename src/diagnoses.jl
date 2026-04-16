# src/diagnoses.jl
# Clinical diagnoses derived from computed metrics

# ---------------------------------------------------------------------------
# Confidence helper
# ---------------------------------------------------------------------------

# Returns "definite", "borderline", or "not_applicable"
function _confidence(value, threshold, direction::Symbol; borderline_pct = 0.10)
    margin = abs(threshold) * borderline_pct
    if direction == :above
        # "definite" if value is clearly above threshold
        if value >= threshold + margin
            return "definite"
        elseif value >= threshold - margin
            return "borderline"
        else
            return "not_applicable"
        end
    elseif direction == :below
        if value <= threshold - margin
            return "definite"
        elseif value <= threshold + margin
            return "borderline"
        else
            return "not_applicable"
        end
    end
    return "not_applicable"
end

const DISCLAIMER = "Based on simulation output, not a clinical measurement."

# ---------------------------------------------------------------------------
# Diagnoses
# ---------------------------------------------------------------------------

function compute_diagnoses(metrics::Dict, cfg::Dict)
    scenario = cfg["scenario"]
    diagnoses = Dict[]

    gpl = metrics["plasma_glucose"]
    der = metrics["derived"]

    fasting    = gpl["fasting_mmol_L"]
    two_hour   = gpl["two_hour_value_mmol_L"]
    homa_ir    = der["homa_ir"]
    ii         = der["insulinogenic_index"]   # may be nothing

    # -----------------------------------------------------------------------
    # 1. Glucose tolerance
    # -----------------------------------------------------------------------
    is_ogtt = scenario in ["ogtt3", "ogtt4"]

    # T2D check first (highest priority)
    t2d_fasting_conf = _confidence(fasting, 7.0, :above)
    t2d_2h_conf      = is_ogtt ? _confidence(two_hour, 11.1, :above) : "not_applicable"
    t2d_conf = (t2d_fasting_conf == "definite" || t2d_2h_conf == "definite") ? "definite" :
               (t2d_fasting_conf == "borderline" || t2d_2h_conf == "borderline") ? "borderline" :
               "not_applicable"

    if t2d_conf in ["definite", "borderline"]
        evidence = DISCLAIMER * " Fasting glucose: $(round(fasting, digits=1)) mmol/L."
        if is_ogtt
            evidence *= " 2-hour glucose: $(round(two_hour, digits=1)) mmol/L."
        end
        push!(diagnoses, Dict(
            "ontology_term_code" => "44054006",
            "name"               => "Type 2 Diabetes Mellitus Risk",
            "present"            => true,
            "confidence"         => t2d_conf,
            "evidence"           => evidence,
        ))
    else
        # IFG: fasting 5.6 – 6.9
        ifg_lower_conf = _confidence(fasting, 5.6, :above)
        ifg_upper_conf = _confidence(fasting, 7.0, :below)
        ifg_conf = (ifg_lower_conf in ["definite","borderline"] && ifg_upper_conf in ["definite","borderline"]) ?
                    (ifg_lower_conf == "definite" && ifg_upper_conf == "definite" ? "definite" : "borderline") :
                    "not_applicable"

        if ifg_conf in ["definite", "borderline"]
            push!(diagnoses, Dict(
                "ontology_term_code" => "HDT-DIAG-ELEVATED-FASTING-GLUCOSE",
                "name"               => "Elevated Fasting Glucose",
                "present"            => true,
                "confidence"         => ifg_conf,
                "evidence"           => DISCLAIMER * " Fasting glucose: $(round(fasting, digits=1)) mmol/L.",
            ))
        end

        # IGT: 2h 7.8 – 11.0 (OGTT only)
        if is_ogtt
            igt_lower_conf = _confidence(two_hour, 7.8, :above)
            igt_upper_conf = _confidence(two_hour, 11.1, :below)
            igt_conf = (igt_lower_conf in ["definite","borderline"] && igt_upper_conf in ["definite","borderline"]) ?
                        (igt_lower_conf == "definite" && igt_upper_conf == "definite" ? "definite" : "borderline") :
                        "not_applicable"

            if igt_conf in ["definite", "borderline"]
                push!(diagnoses, Dict(
                    "ontology_term_code" => "9414007",
                    "name"               => "Impaired Glucose Tolerance",
                    "present"            => true,
                    "confidence"         => igt_conf,
                    "evidence"           => DISCLAIMER * " 2-hour glucose: $(round(two_hour, digits=1)) mmol/L.",
                ))
            end
        end

    end

    # -----------------------------------------------------------------------
    # 2. Insulin resistance (HOMA-IR)
    # -----------------------------------------------------------------------
    if homa_ir > 3.0 * 1.1
        ir_code = "HDT-DIAG-INSULIN-RESISTANCE"
        ir_name = "Insulin Resistance"
        ir_conf = "definite"
    elseif homa_ir > 3.0 * 0.9
        ir_code = "HDT-DIAG-INSULIN-RESISTANCE"
        ir_name = "Insulin Resistance"
        ir_conf = "borderline"
    elseif homa_ir > 2.0 * 1.1
        ir_code = "HDT-DIAG-BORDERLINE-IR"
        ir_name = "Borderline Insulin Resistance"
        ir_conf = "definite"
    elseif homa_ir > 2.0 * 0.9
        ir_code = "HDT-DIAG-BORDERLINE-IR"
        ir_name = "Borderline Insulin Resistance"
        ir_conf = "borderline"
    else
        ir_code = nothing
        ir_name = ""
        ir_conf = ""
    end

    if ir_code !== nothing
        push!(diagnoses, Dict(
            "ontology_term_code" => ir_code,
            "name"               => ir_name,
            "present"            => true,
            "confidence"         => ir_conf,
            "evidence"           => DISCLAIMER * " HOMA-IR: $(round(homa_ir, digits=2)).",
        ))
    end

    # -----------------------------------------------------------------------
    # 3. Beta-cell function (insulinogenic index) — OGTT only
    # -----------------------------------------------------------------------
    if is_ogtt && ii !== nothing
        if ii >= 0.4 * 0.9
            bcf_code = nothing
            bcf_name = ""
            bcf_conf = ""
        elseif ii >= 0.2 * 0.9
            bcf_code = "HDT-DIAG-REDUCED-BCF"
            bcf_name = "Reduced Beta-cell Function"
            bcf_conf = (ii < 0.4 * 1.1 && ii > 0.2 * 0.9) ? "definite" : "borderline"
        else
            bcf_code = "HDT-DIAG-IMPAIRED-BCF"
            bcf_name = "Impaired Beta-cell Function"
            bcf_conf = ii < 0.2 * 0.9 ? "definite" : "borderline"
        end

        bcf_code !== nothing && push!(diagnoses, Dict(
            "ontology_term_code" => bcf_code,
            "name"               => bcf_name,
            "present"            => true,
            "confidence"         => bcf_conf,
            "evidence"           => DISCLAIMER * " Insulinogenic index: $(round(ii, digits=3)).",
        ))
    end

    return diagnoses
end

# ---------------------------------------------------------------------------
# Advices — derived from active diagnoses, matched to ontology triggers
# ---------------------------------------------------------------------------

function compute_advices(diagnoses::Vector)
    active_codes = Set(d["ontology_term_code"] for d in diagnoses if get(d, "present", false))
    advices = Dict[]

    if "9414007" in active_codes
        push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-MONITOR-GLUCOSE",   "name" => "Increase Glucose Monitoring Frequency", "present" => true))
        push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-INCREASE-ACTIVITY", "name" => "Increase Physical Activity",             "present" => true))
    end
    if "44054006" in active_codes
        push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-REDUCE-CARBS",      "name" => "Reduce Carbohydrate Intake",             "present" => true))
        push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-CONSULT-HCP",       "name" => "Consult Healthcare Provider",            "present" => true))
    end
    if "HDT-DIAG-INSULIN-RESISTANCE" in active_codes
        push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-LIFESTYLE-IR",      "name" => "Lifestyle Intervention for Insulin Resistance", "present" => true))
    end
    if "HDT-DIAG-IMPAIRED-BCF" in active_codes
        push!(advices, Dict("ontology_term_code" => "HDT-ADVICE-BETA-CELL-FOLLOWUP","name" => "Beta-cell Function Follow-up",           "present" => true))
    end

    return advices
end
