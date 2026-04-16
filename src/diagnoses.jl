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
# Main function
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
    tir        = gpl["time_in_range_3p9_to_10_percent"]

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
        criteria_vals = Dict("fasting_mmol_L" => fasting)
        if is_ogtt
            criteria_vals["two_hour_mmol_L"] = two_hour
        end
        push!(diagnoses, Dict(
            "name"              => "Type 2 Diabetes (simulated flag)",
            "code"              => "T2D",
            "category"          => "glucose_tolerance",
            "ontology_ref"      => "diagnoses.glucose_tolerance.T2D",
            "confidence"        => t2d_conf,
            "disclaimer"        => DISCLAIMER,
            "criteria_values"   => criteria_vals,
            "supporting_metrics"=> ["fasting_mmol_L", is_ogtt ? "two_hour_value_mmol_L" : nothing],
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
                "name"              => "Impaired Fasting Glucose",
                "code"              => "IFG",
                "category"          => "glucose_tolerance",
                "ontology_ref"      => "diagnoses.glucose_tolerance.IFG",
                "confidence"        => ifg_conf,
                "disclaimer"        => DISCLAIMER,
                "criteria_values"   => Dict("fasting_mmol_L" => fasting),
                "supporting_metrics"=> ["fasting_mmol_L"],
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
                    "name"              => "Impaired Glucose Tolerance",
                    "code"              => "IGT",
                    "category"          => "glucose_tolerance",
                    "ontology_ref"      => "diagnoses.glucose_tolerance.IGT",
                    "confidence"        => igt_conf,
                    "disclaimer"        => DISCLAIMER,
                    "criteria_values"   => Dict("two_hour_mmol_L" => two_hour),
                    "supporting_metrics"=> ["two_hour_value_mmol_L"],
                ))
            end
        end

        # NGT: fasting < 5.6 AND (2h < 7.8 or not OGTT)
        ngt_fasting_conf = _confidence(fasting, 5.6, :below)
        ngt_2h_conf      = is_ogtt ? _confidence(two_hour, 7.8, :below) : "definite"
        ngt_conf = (ngt_fasting_conf in ["definite","borderline"] && ngt_2h_conf in ["definite","borderline"]) ?
                    (ngt_fasting_conf == "definite" && ngt_2h_conf == "definite" ? "definite" : "borderline") :
                    "not_applicable"

        if ngt_conf in ["definite", "borderline"]
            criteria_vals = Dict("fasting_mmol_L" => fasting)
            if is_ogtt
                criteria_vals["two_hour_mmol_L"] = two_hour
            end
            push!(diagnoses, Dict(
                "name"              => "Normal Glucose Tolerance",
                "code"              => "NGT",
                "category"          => "glucose_tolerance",
                "ontology_ref"      => "diagnoses.glucose_tolerance.NGT",
                "confidence"        => ngt_conf,
                "disclaimer"        => DISCLAIMER,
                "criteria_values"   => criteria_vals,
                "supporting_metrics"=> ["fasting_mmol_L", is_ogtt ? "two_hour_value_mmol_L" : nothing],
            ))
        end
    end

    # -----------------------------------------------------------------------
    # 2. Insulin resistance (HOMA-IR)
    # -----------------------------------------------------------------------
    if homa_ir > 3.0 * 1.1   # clearly > 3.0
        ir_code = "Insulin_Resistant"
        ir_name = "Insulin Resistance"
        ir_conf = "definite"
    elseif homa_ir > 3.0 * 0.9   # borderline around 3.0
        ir_code = "Insulin_Resistant"
        ir_name = "Insulin Resistance"
        ir_conf = "borderline"
    elseif homa_ir > 2.0 * 1.1
        ir_code = "Borderline_IR"
        ir_name = "Borderline Insulin Resistance"
        ir_conf = "definite"
    elseif homa_ir > 2.0 * 0.9
        ir_code = "Borderline_IR"
        ir_name = "Borderline Insulin Resistance"
        ir_conf = "borderline"
    else
        ir_code = "Normal_IR"
        ir_name = "Normal Insulin Sensitivity"
        ir_conf = _confidence(homa_ir, 2.0, :below) == "not_applicable" ? "definite" :
                   _confidence(homa_ir, 2.0, :below)
        ir_conf = "definite"  # clearly below 2.0
    end

    push!(diagnoses, Dict(
        "name"              => ir_name,
        "code"              => ir_code,
        "category"          => "insulin_resistance",
        "ontology_ref"      => "diagnoses.insulin_resistance.$(ir_code)",
        "confidence"        => ir_conf,
        "disclaimer"        => DISCLAIMER,
        "criteria_values"   => Dict("homa_ir" => homa_ir),
        "supporting_metrics"=> ["homa_ir"],
    ))

    # -----------------------------------------------------------------------
    # 3. Beta-cell function (insulinogenic index) — OGTT only
    # -----------------------------------------------------------------------
    if is_ogtt && ii !== nothing
        if ii >= 0.4 * 0.9
            bcf_code = "Normal_BCF"
            bcf_name = "Normal Beta-cell Function"
            bcf_conf = ii >= 0.4 * 1.1 ? "definite" : "borderline"
        elseif ii >= 0.2 * 0.9
            bcf_code = "Reduced_BCF"
            bcf_name = "Reduced Beta-cell Function"
            bcf_conf = (ii < 0.4 * 1.1 && ii > 0.2 * 0.9) ? "definite" : "borderline"
        else
            bcf_code = "Impaired_BCF"
            bcf_name = "Impaired Beta-cell Function"
            bcf_conf = ii < 0.2 * 0.9 ? "definite" : "borderline"
        end

        push!(diagnoses, Dict(
            "name"              => bcf_name,
            "code"              => bcf_code,
            "category"          => "beta_cell_function",
            "ontology_ref"      => "diagnoses.beta_cell_function.$(bcf_code)",
            "confidence"        => bcf_conf,
            "disclaimer"        => DISCLAIMER,
            "criteria_values"   => Dict("insulinogenic_index" => ii),
            "supporting_metrics"=> ["insulinogenic_index"],
        ))
    end

    # -----------------------------------------------------------------------
    # 4. Glycemic control (TIR) — CGM only
    # -----------------------------------------------------------------------
    if scenario == "cgm"
        if tir >= 70.0 * 0.9
            gc_code = "Good_Control"
            gc_name = "Good Glycemic Control"
            gc_conf = tir >= 70.0 * 1.1 ? "definite" : "borderline"
        elseif tir >= 50.0 * 0.9
            gc_code = "Suboptimal_Control"
            gc_name = "Suboptimal Glycemic Control"
            gc_conf = (tir < 70.0 * 1.1 && tir > 50.0 * 0.9) ? "definite" : "borderline"
        else
            gc_code = "Poor_Control"
            gc_name = "Poor Glycemic Control"
            gc_conf = tir < 50.0 * 0.9 ? "definite" : "borderline"
        end

        push!(diagnoses, Dict(
            "name"              => gc_name,
            "code"              => gc_code,
            "category"          => "glycemic_control",
            "ontology_ref"      => "diagnoses.glycemic_control.$(gc_code)",
            "confidence"        => gc_conf,
            "disclaimer"        => DISCLAIMER,
            "criteria_values"   => Dict("time_in_range_percent" => tir),
            "supporting_metrics"=> ["time_in_range_3p9_to_10_percent"],
        ))
    end

    return diagnoses
end
