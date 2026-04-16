# src/metrics.jl
# Clinical metrics computed from ODE solution
using Statistics: mean

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function trapz(t::AbstractVector, y::AbstractVector)
    n = length(t)
    n < 2 && return 0.0
    s = 0.0
    for i in 1:(n - 1)
        s += 0.5 * (y[i] + y[i + 1]) * (t[i + 1] - t[i])
    end
    return s
end

function find_peak(t::AbstractVector, y::AbstractVector)
    idx = argmax(y)
    return (y[idx], t[idx])
end

function interpolate_at(sol, times, idx::Int)
    return [sol(Float64(t))[idx] for t in times]
end

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

function compute_metrics(sol, data, cfg, final_params)
    t_sol = sol.t
    Gb = cfg["Gb"]     # fasting plasma glucose (mmol/L)
    Ib = cfg["Ib"]     # fasting plasma insulin (mU/L)
    scenario = cfg["scenario"]

    # Full solution traces
    Ggut_trace = sol[1, :]
    Gpl_trace  = sol[2, :]
    Ipl_trace  = sol[3, :]
    Irem_trace = sol[4, :]

    # -----------------------------------------------------------------------
    # gut_glucose metrics
    # -----------------------------------------------------------------------
    peak_ggut, tpeak_ggut = find_peak(t_sol, Ggut_trace)
    auc_ggut = trapz(t_sol, Ggut_trace)

    # absorption_complete: first t after peak where Ggut < 10% of peak
    absorption_complete = nothing
    threshold_ggut = 0.1 * peak_ggut
    for i in 2:length(t_sol)
        if t_sol[i] >= tpeak_ggut && Ggut_trace[i] < threshold_ggut
            absorption_complete = t_sol[i]
            break
        end
    end

    gut_glucose_metrics = Dict(
        "ontology_ref"                 => "state_variables.gut_glucose",
        "peak_mg"                      => peak_ggut,
        "time_to_peak_min"             => tpeak_ggut,
        "auc_mg_min"                   => auc_ggut,
        "absorption_complete_pct10_min"=> absorption_complete,
    )

    # -----------------------------------------------------------------------
    # plasma_glucose metrics
    # -----------------------------------------------------------------------
    peak_gpl, tpeak_gpl = find_peak(t_sol, Gpl_trace)
    auc_gpl  = trapz(t_sol, Gpl_trace)
    iauc_gpl = trapz(t_sol, max.(Gpl_trace .- Gb, 0.0))

    # TIR: 3.9 – 10.0 mmol/L
    in_range_time  = 0.0
    above_range_time = 0.0
    below_range_time = 0.0
    total_time = t_sol[end] - t_sol[1]
    for i in 1:(length(t_sol) - 1)
        dt = t_sol[i + 1] - t_sol[i]
        g_mid = 0.5 * (Gpl_trace[i] + Gpl_trace[i + 1])
        if g_mid >= 3.9 && g_mid <= 10.0
            in_range_time += dt
        elseif g_mid > 10.0
            above_range_time += dt
        else
            below_range_time += dt
        end
    end
    tir_pct   = total_time > 0 ? 100.0 * in_range_time / total_time : 0.0
    tar_pct   = total_time > 0 ? 100.0 * above_range_time / total_time : 0.0
    tbr_pct   = total_time > 0 ? 100.0 * below_range_time / total_time : 0.0

    # return_to_near_baseline: first t after peak where Gpl < Gb + 0.5
    return_to_baseline = nothing
    for i in 2:length(t_sol)
        if t_sol[i] >= tpeak_gpl && Gpl_trace[i] < Gb + 0.5
            return_to_baseline = t_sol[i]
            break
        end
    end

    # 2h value via dense interpolation
    two_hour_val = sol(120.0)[2]

    plasma_glucose_metrics = Dict(
        "ontology_ref"                       => "state_variables.plasma_glucose",
        "fasting_mmol_L"                     => Gb,
        "peak_mmol_L"                        => peak_gpl,
        "time_to_peak_min"                   => tpeak_gpl,
        "auc_mmol_L_min"                     => auc_gpl,
        "iauc_mmol_L_min"                    => iauc_gpl,
        "excursion_mmol_L"                   => peak_gpl - Gb,
        "time_in_range_3p9_to_10_percent"    => tir_pct,
        "time_above_range_percent"           => tar_pct,
        "time_below_range_percent"           => tbr_pct,
        "return_to_near_baseline_min"        => return_to_baseline,
        "two_hour_value_mmol_L"              => two_hour_val,
    )

    # -----------------------------------------------------------------------
    # plasma_insulin metrics
    # -----------------------------------------------------------------------
    peak_ipl, tpeak_ipl = find_peak(t_sol, Ipl_trace)
    auc_ipl  = trapz(t_sol, Ipl_trace)
    iauc_ipl = trapz(t_sol, max.(Ipl_trace .- Ib, 0.0))

    # Phase AUCs via interpolation on fine grid
    t_phase1 = filter(t -> t <= 30.0, t_sol)
    t_phase2 = filter(t -> t >= 30.0 && t <= 120.0, t_sol)
    if isempty(t_phase1) || length(t_phase1) < 2
        auc_phase1 = 0.0
    else
        auc_phase1 = trapz(t_phase1, interpolate_at(sol, t_phase1, 3))
    end
    if isempty(t_phase2) || length(t_phase2) < 2
        auc_phase2 = 0.0
    else
        auc_phase2 = trapz(t_phase2, interpolate_at(sol, t_phase2, 3))
    end

    plasma_insulin_metrics = Dict(
        "ontology_ref"              => "state_variables.plasma_insulin",
        "fasting_mU_L"              => Ib,
        "peak_mU_L"                 => peak_ipl,
        "time_to_peak_min"          => tpeak_ipl,
        "auc_mU_L_min"              => auc_ipl,
        "iauc_mU_L_min"             => iauc_ipl,
        "first_phase_auc_mU_L_min"  => auc_phase1,
        "second_phase_auc_mU_L_min" => auc_phase2,
    )

    # -----------------------------------------------------------------------
    # interstitium_insulin metrics
    # -----------------------------------------------------------------------
    peak_irem, tpeak_irem = find_peak(t_sol, Irem_trace)
    auc_irem = trapz(t_sol, Irem_trace)

    interstitium_insulin_metrics = Dict(
        "ontology_ref"     => "state_variables.interstitium_insulin",
        "peak_mU_L"        => peak_irem,
        "time_to_peak_min" => tpeak_irem,
        "auc_mU_L_min"     => auc_irem,
    )

    # -----------------------------------------------------------------------
    # Derived / composite metrics
    # -----------------------------------------------------------------------

    # HOMA-IR: (fasting glucose mmol/L × fasting insulin mU/L) / 22.5
    homa_ir = (Gb * Ib) / 22.5

    # Insulinogenic index: (I30 - Ib) / (G30 - Gb)
    # Only meaningful for OGTT scenarios
    insulinogenic_index = nothing
    if scenario in ["ogtt3", "ogtt4"]
        I30 = sol(30.0)[3]
        G30 = sol(30.0)[2]
        dG30 = G30 - Gb
        if abs(dG30) > 1e-6
            insulinogenic_index = (I30 - Ib) / dG30
        end
    end

    # ISI Matsuda approximation
    # 10000 / sqrt(Gb_mgdL × Ib × Gmean_mgdL × Imean)
    # Convert Gb from mmol/L to mg/dL: × 18.0
    Gb_mgdL   = Gb * 18.0
    Gmean_mgdL = mean(Gpl_trace) * 18.0
    Imean      = mean(Ipl_trace)
    isi_denom  = sqrt(Gb_mgdL * Ib * Gmean_mgdL * Imean)
    isi_matsuda = isi_denom > 0 ? 10000.0 / isi_denom : nothing

    # Disposition index
    disposition_index = nothing
    if insulinogenic_index !== nothing && isi_matsuda !== nothing
        disposition_index = isi_matsuda * insulinogenic_index
    end

    derived_metrics = Dict(
        "homa_ir"               => homa_ir,
        "insulinogenic_index"   => insulinogenic_index,
        "isi_matsuda_approx"    => isi_matsuda,
        "disposition_index"     => disposition_index,
    )

    return Dict(
        "gut_glucose"            => gut_glucose_metrics,
        "plasma_glucose"         => plasma_glucose_metrics,
        "plasma_insulin"         => plasma_insulin_metrics,
        "interstitium_insulin"   => interstitium_insulin_metrics,
        "derived"                => derived_metrics,
    )
end
