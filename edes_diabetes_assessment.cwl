class: Workflow
cwlVersion: "HDT-CWL-1.1"
label: "EDES OGTT Diabetes Assessment"
doc: >
  Single-step pipeline that accepts OGTT glucose and insulin time series,
  runs the EDES ODE model (ogtt4 scenario), and returns simulated trajectories,
  fitted metabolic parameters, and clinical risk classifications
  (glucose tolerance, insulin resistance, beta-cell function).

inputs:
  scenario:
    type: string
    ontologyTermCode: "HDT-EDES-SCENARIO"
    doc: "EDES scenario selector; ogtt4 only in v2.0.0"
  plasma_glucose:
    type: object
    ontologyTermCode: "14749-6"
    doc: "OGTT plasma glucose time series {timestamps_min, values} in mmol/L"
  fasting_insulin:
    type: float
    ontologyTermCode: "20448-7"
    doc: "Fasting plasma insulin baseline in mU/L"
  body_weight:
    type: float
    ontologyTermCode: "29463-7"
    doc: "Patient body weight in kg"
  pretrained_params:
    type: object
    ontologyTermCode: "HDT-EDES-PARAMS"
    doc: "Optional pre-fitted parameters {k1,k5,k6,k8,sigma_g,sigma_i}; skips optimisation if provided"
  plasma_insulin:
    type: object
    ontologyTermCode: "HDT-EDES-INSULIN"
    doc: "OGTT plasma insulin time series {timestamps_min, values} in mU/L; required for ogtt4"

outputs:
  plasma_glucose_sim:
    type: object
    outputSource: edes_model/plasma_glucose_sim
    ontologyTermCode: "HDT-EDES-PLASMA-GLUCOSE"
    doc: "Simulated plasma glucose trajectory (mmol/L)"
  plasma_insulin_sim:
    type: object
    outputSource: edes_model/plasma_insulin_sim
    ontologyTermCode: "HDT-EDES-PLASMA-INSULIN"
    doc: "Simulated plasma insulin trajectory (mU/L)"
  gut_glucose:
    type: object
    outputSource: edes_model/gut_glucose
    ontologyTermCode: "HDT-EDES-GUT-GLUCOSE"
    doc: "Gut glucose absorption over time (mg)"
  interstitium_insulin:
    type: object
    outputSource: edes_model/interstitium_insulin
    ontologyTermCode: "HDT-EDES-INTERSTITIUM-INSULIN"
    doc: "Interstitium insulin trajectory (mU/L)"
  fit_params:
    type: object
    outputSource: edes_model/fit_params
    ontologyTermCode: "HDT-EDES-FIT-PARAMS"
    doc: "Fitted or supplied EDES parameters {k1,k5,k6,k8,sigma_g,sigma_i}"
  igt:
    type: boolean
    outputSource: edes_model/igt
    ontologyTermCode: "9414007"
    doc: "Impaired Glucose Tolerance flag (peak 2h glucose 7.8-11.1 mmol/L)"
  t2d_risk:
    type: boolean
    outputSource: edes_model/t2d_risk
    ontologyTermCode: "44054006"
    doc: "Type 2 Diabetes Mellitus risk flag (peak 2h glucose >= 11.1 mmol/L)"
  elevated_fasting_glucose:
    type: boolean
    outputSource: edes_model/elevated_fasting_glucose
    ontologyTermCode: "HDT-DIAG-ELEVATED-FASTING-GLUCOSE"
    doc: "Elevated fasting glucose flag (fasting > 6.1 mmol/L)"
  normal_glucose:
    type: boolean
    outputSource: edes_model/normal_glucose
    ontologyTermCode: "HDT-DIAG-NORMAL-GLUCOSE"
    doc: "Normal glucose regulation flag"
  insulin_resistance:
    type: boolean
    outputSource: edes_model/insulin_resistance
    ontologyTermCode: "HDT-DIAG-INSULIN-RESISTANCE"
    doc: "Insulin resistance flag (HOMA-IR > 3.0)"
  borderline_ir:
    type: boolean
    outputSource: edes_model/borderline_ir
    ontologyTermCode: "HDT-DIAG-BORDERLINE-IR"
    doc: "Borderline insulin resistance flag (HOMA-IR 2.0-3.0)"
  normal_ir:
    type: boolean
    outputSource: edes_model/normal_ir
    ontologyTermCode: "HDT-DIAG-NORMAL-IR"
    doc: "Normal insulin sensitivity flag (HOMA-IR < 2.0)"
  impaired_bcf:
    type: boolean
    outputSource: edes_model/impaired_bcf
    ontologyTermCode: "HDT-DIAG-IMPAIRED-BCF"
    doc: "Impaired beta-cell function flag (insulinogenic_index < 0.2)"
  reduced_bcf:
    type: boolean
    outputSource: edes_model/reduced_bcf
    ontologyTermCode: "HDT-DIAG-REDUCED-BCF"
    doc: "Reduced beta-cell function flag (insulinogenic_index 0.2-0.4)"
  normal_bcf:
    type: boolean
    outputSource: edes_model/normal_bcf
    ontologyTermCode: "HDT-DIAG-NORMAL-BCF"
    doc: "Normal beta-cell function flag (insulinogenic_index >= 0.4)"
  consult_hcp:
    type: boolean
    outputSource: edes_model/consult_hcp
    ontologyTermCode: "HDT-ADVICE-CONSULT-HCP"
    doc: "Advice: consult healthcare provider (triggered on T2D risk)"
  reduce_carbs:
    type: boolean
    outputSource: edes_model/reduce_carbs
    ontologyTermCode: "HDT-ADVICE-REDUCE-CARBS"
    doc: "Advice: reduce carbohydrate intake (triggered on T2D risk)"
  increase_activity:
    type: boolean
    outputSource: edes_model/increase_activity
    ontologyTermCode: "HDT-ADVICE-INCREASE-ACTIVITY"
    doc: "Advice: increase physical activity (triggered on IGT)"
  monitor_glucose:
    type: boolean
    outputSource: edes_model/monitor_glucose
    ontologyTermCode: "HDT-ADVICE-MONITOR-GLUCOSE"
    doc: "Advice: increase glucose monitoring frequency (triggered on IGT)"
  lifestyle_ir:
    type: boolean
    outputSource: edes_model/lifestyle_ir
    ontologyTermCode: "HDT-ADVICE-LIFESTYLE-IR"
    doc: "Advice: lifestyle intervention for insulin resistance (triggered on HOMA-IR > 3.0)"
  beta_cell_followup:
    type: boolean
    outputSource: edes_model/beta_cell_followup
    ontologyTermCode: "HDT-ADVICE-BETA-CELL-FOLLOWUP"
    doc: "Advice: beta-cell function follow-up (triggered on insulinogenic_index < 0.2)"

steps:
  edes_model:
    run: "edes-universal:2.0.4"
    in:
      scenario:          scenario
      plasma_glucose:    plasma_glucose
      fasting_insulin:   fasting_insulin
      body_weight:       body_weight
      pretrained_params: pretrained_params
      plasma_insulin:    plasma_insulin
    out:
      - plasma_glucose_sim
      - plasma_insulin_sim
      - gut_glucose
      - interstitium_insulin
      - fit_params
      - igt
      - t2d_risk
      - elevated_fasting_glucose
      - normal_glucose
      - insulin_resistance
      - borderline_ir
      - normal_ir
      - impaired_bcf
      - reduced_bcf
      - normal_bcf
      - consult_hcp
      - reduce_carbs
      - increase_activity
      - monitor_glucose
      - lifestyle_ir
      - beta_cell_followup
