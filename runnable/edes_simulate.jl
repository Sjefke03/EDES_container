# edes_simulate.jl
# Standalone runnable script — exact translation of EDES_Model_Identifiability.ipynb
# Run from the repo root: julia runnable/edes_simulate.jl

# ─── Cell 1: Package setup ────────────────────────────────────────────────────
import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()

# ─── Cell 2: Load libraries ───────────────────────────────────────────────────
using OrdinaryDiffEq
using CairoMakie
using Random
using QuasiMonteCarlo
using Optimization
using OptimizationOptimJL
using Distributions
include(joinpath(@__DIR__, "..", "utils", "SelectionMethods.jl"))

# ─── Cell 3: ODE function definition ─────────────────────────────────────────
"""
edesode!(du, u, p, t)
  This function defines the system of ordinary differential equations (ODEs) that describe the EDES model.
"""
function edesode!(du, u, p, t)
  #specify initial values
    Ggut, Gpl, Ipl, Irem = u
    #specify the parameters
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib = p

    # Equation defining rate of decay of ingested glucose in stomach/gut
    du[1] = sigma * k1^sigma * t^(sigma-1) * exp(-(k1*t)^sigma) * Dmeal - k2 * Ggut

    # Equation describing the rate of change of plasma glucose
    gliv = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem #Endogenous glucose production in liver
    ggut = k2 * (f / (Vg * bw)) * Ggut #glucose appearing from meal
    u_ii =  EGPb * ((Km + Gb)/Gb) * (Gpl / (Km + Gpl)) # insulin independent uptake of glucose to peripheral tissues
    u_id = k5 * beta * Irem * (Gpl / (Km + Gpl)) # insulin dependent uptake of glucose to peripheral tissues
    u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren) # renal excretion of excess glucose (>Gren threshold)

    du[2]  = gliv + ggut - u_ii - u_id - u_ren #full glucose in plasma equation

    #  Equation describing the rate of change of plasma insulin
    i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * Gb + k8 * tau_d * du[2]) #rate of insulin production in pancreas (funtion of glucose)
    i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib) #rate of insulin degredation by liver
    i_int = k9 * (Ipl - Ib) #rate of insulin transport to interstitial compartment
    du[3] = i_pnc - i_liv - i_int # full insulin in plasma equation
    du[4] = i_int - k10 * Irem #full insulin in instertital compartment equation
end

# ─── Cell 4: Parameter setup and ODE problem definition ──────────────────────
# specify physiology parameters
bw = 70.0 #body weight (kg)
Gb = 5.0 #fasting glucose (mmol/l)
Ib = 10.0 #fasting insulin (uIU/ml)

# meal parameters
Dmeal = 75.0e3 #amount of glucose in meal (mg)

# specify the values for the EDES model parameters
k1 = 0.0105   # rate of glucose decay in stomach
k2 = 0.28     # rate of glucose transport from stomach to plasma
k3 = 6.07e-3  # rate of glucose effect on endogenous glucose production
k4 = 2.35e-4  # rate of insulin effect on endogenous glucose production
k5 = 0.0424   # rate of insulin dependent glucose uptake into peripheral tissues
k6 = 2.2975   # rate of insulin secretion (proportional to Gb)
k7 = 1.15     # rate of insulin secretion (integral of Gpl)
k8 = 7.27     # rate of insulin secretion (derivative of Gpl)
k9 = 3.83e-2  # delay parameter plasma to insterstitial insulin
k10 = 2.84e-1 # rate constant for degredation of insulin in remote compartment
tau_i = 31.0  # time delay integrator function
tau_d = 3.0   # time delay
beta = 1.0    # conversion factor insulin
Gren = 9.0    # threshold for renal excretion of glucose
EGPb = 0.043  # basal rate of endogenous glucose production
Km = 13.2     # Michealis Menten coeficient for glucose uptake into periphery
f = 0.005551  #
Vg = 17.0 / bw # volume of distribution for glucose (function of body weight)
c1 = 0.1      # model constant
sigma = 1.4   # shape factor (appearance of meal)

# define the collection of parameters
p = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]

# set initial conditions for each state variable
# [glucose mass in gut, glcuose concentration in plasma, insulin concentration in plasma, insulin constration in instertitium]
u0 = [0.0, Gb, Ib, 0.0]

# set time span for model simulation
tspan = (0.0, 240.0)

# define the ODE problem
prob = ODEProblem(edesode!, u0, tspan, p)

# ─── Cell 5: Solve and visualize all 4 state variables ───────────────────────
#solve the ODE problem using the solve function
solution = solve(prob)

# visualize the solution
solution_figure = let f = Figure(size=(500,500))

    ax_g_gut = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg]", title="Gut Glucose")
    ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
    ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
    ax_i_int = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

    lines!(ax_g_gut, solution.t, solution[1,:], color=Makie.wong_colors()[1])
    lines!(ax_g_plasma, solution.t, solution[2,:], color=Makie.wong_colors()[1])
    lines!(ax_i_plasma, solution.t, solution[3,:], color=Makie.wong_colors()[1])
    lines!(ax_i_int, solution.t, solution[4,:], color=Makie.wong_colors()[1])

    f
end
save(joinpath(@__DIR__, "fig_cell5_forward_sim.png"), solution_figure)

# ─── Cell 6: construct_parameters function ────────────────────────────────────
#define a function that will specify parameters to be estimated and take in values for fixed parameters
function construct_parameters(θ, c)

    # Estimated parameters
    k1 = θ[1]     # rate og glucose decay in stomach
    k5 = θ[2]     # rate of insulin dependent glucose uptake into peripheral tissues
    k6 = θ[3]     # rate of insulin secretion (proportional to Gb)
    k8 = θ[4]     # rate of insulin secretion (derivative of Gpl) #to fix the value for k8 you can change the value here (an example is provided below)

    # Fixed parameters
    k2 = c[1]     # rate of glcuose transport from stomach to plasma
    k3 = c[2]     # rate of glucose effect on endogenous glucose production
    k4 = c[3]     # rate of insulin effect on endogenous glucose production
    k7 = c[4]     # rate of insulin secretion (integral of Gpl)
    k9 = c[5]     # delay parameter plasma to insterstitial insulin
    k10 = c[6]    # rate constant for degredation of insulin in remote compartment
    tau_i = c[7]  # time delay integrator function
    tau_d = c[8]  # time delay
    beta = c[9]   # conversion factor insulin
    Gren = c[10]  # threshold for renal excretion of glucose
    EGPb = c[11]  # basal rate of endogenous glucose production
    Km = c[12]    # Michealis Menten coeficient for glucose uptake into periphery
    f = c[13]     #
    Vg = c[14]    # volume of distribution for glucose (function of body weight)
    c1 = c[15]    # model constant
    sigma = c[16] # shape factor (appearance of meal)
    Dmeal = c[17] #amount of glucose in meal (mg)
    bw = c[18]    #body weight (kg)
    Gb = c[19]    #fasting glucose (mmol/l)
    Ib = c[20]    #fasting insulin (uIU/ml)
    return [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

# ─── Cell 7: Loss function ────────────────────────────────────────────────────
#define our loss function
#
#loss function takes in values for the estimated parameter and a tuple containing the ODE problem name, values for the fixed parameters and model constants, and the measured experimetnal data.
function loss(θ,(problem, constants ,data))
  #measured glucose and insulin data, including the sampling time points
    glucose_data = data[2,:]
    insulin_data = data[3,:]
    data_timepoints = data[1,:]

    #generate the full parameter vector by combining current values for the estimated parameters with the fixed values and constants.
    p = construct_parameters(θ, constants)

    # solve the ode problem for these parameter values, in this function we define the parameters as p
    # We also only save the ODE simulation at the timepoints that corrispond to the measured glucose and insulin data so we can calculate the error.
    # we also specify that of the four state varaibles [G_gut,  G_pl, I_pl, I_d] for calculating the error we are only interested in G_pl and I_pl
    # we also specify the initial values (U0) for each state variable ([G_gut,G_pl,I_pl,I_d])  for the ODE solver
    pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2, 3], u0=[0.0, data[2,1], data[3,1], 0.0])
    sol = Array(pred)

    #calcualte the error between the measured glucose and insulin data and the EDES model simulation
    g_loss=(sol[1,:] - glucose_data)/maximum(glucose_data)
    i_loss=(sol[2,:] - insulin_data)/maximum(insulin_data)

    # likelihood for the glucose data
    n = length(glucose_data)
    σ_g = θ[end-1]
    L_g = n*log(σ_g * sqrt(2π)) + 1/(2σ_g^2) * sum(abs2, g_loss)

    # likelihood for the insulin data
    σ_i = θ[end]
    L_i = n*log(σ_i * sqrt(2π)) + 1/(2σ_i^2) * sum(abs2, i_loss)

    # weighted sum of the likelihoods
    return 0.5 * L_g + 0.5 * L_i
end

# ─── Cell 8: Data import and visualization ────────────────────────────────────
#import measured glcuose and insulin data
#meal plasma glucose and insulin values collected during an OGTT
data_glu = [5.4 7.1 8.6 8.9 8.5 7.6 6.6]
data_ins = [7.9 36.7 64.8 75.6 79.6 80.5 68.7]
data_time = [0 15 30 45 60 90 120]
data = [data_time;data_glu;data_ins]
#plasma glucose and insulin values for individual A
#To fit to the data for indivudal A uncomment these lines of code
#data_glu = [4.42 5.1 6.2 6.1 5.1 4.7 4.5]
#data_ins = [4.4 27.3 59.8 78.1 61.0 46.6 40.6]
#data_time = [0 15 30 45 60 90 120]
#data = [data_time;data_glu;data_ins]

#sepecify the fixed parameter values
constants = [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, data_glu[1], data_ins[1]];

# plot of the data
data_figure = let f = Figure(size=(400,200))

    ax_glu = Axis(f[1,1], xlabel="Time [min]", ylabel="Concentration [mM]", title="Glucose Data")
    ax_ins = Axis(f[1,2], xlabel="Time [min]", ylabel="Concentration [mU/L]", title="Insulin Data")

    scatter!(ax_glu, data[1,:], data[2,:], color=Makie.wong_colors()[1])
    scatter!(ax_ins, data[1,:], data[3,:], color=Makie.wong_colors()[1])

    f
end
save(joinpath(@__DIR__, "fig_cell8_data.png"), data_figure)

# ─── Cell 9: Initial guess generation function ────────────────────────────────
function generate_initial_guesses(n_guesses, lower_bounds, upper_bounds; selection_method::SelectionMethod = SelectAll())

  initial_guesses = QuasiMonteCarlo.sample(n_guesses, lower_bounds, upper_bounds, LatinHypercubeSample())

  return evaluate_and_select(initial_guesses, selection_method)

end

# ─── Cell 10: Parameter estimation (1000 initial guesses, 4-parameter model) ──
# calcuate 1000 initial guesses for parameters being estimated using latin-hypercube design
initial_guess = generate_initial_guesses(1000, [0., 0., 0., 0., 1e-4, 1e-4], [1., 0.5, 5., 20., 100.0, 100.0])


results = []

optf = OptimizationFunction(loss, AutoForwardDiff())

# now run the local optimisation for each initial guess
#
# note: here it is not unusual to get errors or warnings as for some combinations of parameters generatined by the generate_initial_guesses function can induce errors in the ODE solver and/or
# the optimisation algorithm by reaching a tolerance that induces a stop. By making use of the try-catch fundtion the loop with catch the error and continute to the next initial guess.
# This is also why we use many initial values, as we expect some parameter combinations to be unsuccessful. If you get an error for every parameter combination remove the try-catch and investigate
# if there is an error in your model code.
for guess in eachcol(initial_guess)
    try
      res = solve(OptimizationProblem(optf, vec(guess), (prob, constants, data), lb = [0,0,0,0, 1e-4, 1e-4], ub=[1,0.5,10,25, 100, 100]), Optimization.LBFGS())
      push!(results, res)
    catch
      continue
    end
end;
println("4-param estimation: $(length(results))/1000 runs succeeded")
if isempty(results)
    # Re-run first guess without try-catch to surface the real error
    error("All optimisation runs failed. Running first guess unguarded to diagnose:\n" *
          sprint(showerror, try
              solve(OptimizationProblem(optf, vec(initial_guess[:,1]), (prob, constants, data),
                    lb=[0,0,0,0,1e-4,1e-4], ub=[1,0.5,10,25,100,100]), Optimization.LBFGS())
              nothing
          catch e; e; end))
end
# find the parameterisation with the lowest cost/objective funtion value
best_index = argmin([r.objective for r in results])
final = results[best_index].u

#solve the ODE problem using the solve function - to get the model simulation for this parameter set
solution = solve(prob, p =construct_parameters(final, constants), u0=[0.0, data[2,1], data[3,1], 0.0]);

# ─── Cell 11: Visualization against data (all 4 state variables) ─────────────

# visualize the solution against the data
solution_figure = let f = Figure(size=(500,500))

    ax_g_gut = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]", title="Gut Glucose")
    ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
    ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
    ax_i_int = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

    lines!(ax_g_gut, solution.t, solution[1,:], color=Makie.wong_colors()[1])
    lines!(ax_g_plasma, solution.t, solution[2,:], color=Makie.wong_colors()[1])
    lines!(ax_i_plasma, solution.t, solution[3,:], color=Makie.wong_colors()[1])
    lines!(ax_i_int, solution.t, solution[4,:], color=Makie.wong_colors()[1])

    band!(ax_g_plasma, solution.t, solution[2,:] .+ final[5], solution[2,:] .- final[5])
    band!(ax_i_plasma, solution.t, solution[3,:] .+ final[6], solution[3,:] .- final[6])

    scatter!(ax_i_plasma, data[1,:], data[3,:], color=Makie.wong_colors()[2])
    scatter!(ax_g_plasma, data[1,:], data[2,:], color=Makie.wong_colors()[2])

    f
end
save(joinpath(@__DIR__, "fig_cell11_fit_4param.png"), solution_figure)

# ─── Cell 12: Parameter distribution histograms ───────────────────────────────
# visualize the histogram of parameter estiamtes for all intiialiaslisations
k1_vals=[]
k5_vals=[]
k6_vals=[]
k8_vals=[]
for r =1:100
  push!(k1_vals,results[r].u[1])
  push!(k5_vals,results[r].u[2])
  push!(k6_vals,results[r].u[3])
  push!(k8_vals,results[r].u[4])
end

solution_figure = let f = Figure(size=(500,500))

    ax_k1 = Axis(f[1,1], xlabel="k1 value", title="k1 distribution")
    ax_k5 = Axis(f[1,2], xlabel="k5 value", title="k5 distribution")
    ax_k6 = Axis(f[2,1], xlabel="k6 value", title="k6 distribution")
    ax_k8 = Axis(f[2,2], xlabel="k8 value", title="k8 distribution")

    hist!(ax_k1, k1_vals, bins=10,color=Makie.wong_colors()[1])
    hist!(ax_k5, k5_vals, bins=10,color=Makie.wong_colors()[1])
    hist!(ax_k6, k6_vals, bins=10,color=Makie.wong_colors()[1])
    hist!(ax_k8, k8_vals, bins=10,color=Makie.wong_colors()[1])

    f
end
save(joinpath(@__DIR__, "fig_cell12_param_hist.png"), solution_figure)

# ─── Cell 13: PLA result struct ───────────────────────────────────────────────
struct PLAResult
    likelihood_values
    parameter_values
    optim_index::Int
    other_parameter_values
end

# ─── Cell 14: Loss function with known sigma and PLA loss constructor ─────────
function loss_known_sigma(θ,(problem, constants ,data, sigma_g, sigma_i))
  #measured glucose and insulin data, including the sampling time points
    glucose_data = data[2,:]
    insulin_data = data[3,:]
    data_timepoints = data[1,:]

    #generate the full parameter vector by combining current values for the estimated parameters with the fixed values and constants.
    p = construct_parameters(θ, constants)

    # solve the ode problem for these parameter values
    pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2, 3], u0=[0.0, data[2,1], data[3,1], 0.0])
    sol = Array(pred)

    #calcualte the error between the measured glucose and insulin data and the EDES model simulation
    g_loss=(sol[1,:] - glucose_data)/maximum(glucose_data)
    i_loss=(sol[2,:] - insulin_data)/maximum(insulin_data)

    # likelihood for the glucose data
    n = length(glucose_data)
    σ_g = sigma_g
    L_g = n*log(σ_g * sqrt(2π)) + 1/(2σ_g^2) * sum(abs2, g_loss)

    # likelihood for the insulin data
    σ_i = sigma_i
    L_i = n*log(σ_i * sqrt(2π)) + 1/(2σ_i^2) * sum(abs2, i_loss)

    # weighted sum of the likelihoods
    return 0.5 * L_g + 0.5 * L_i
end

function loss_pla(loss, pla_param, fixed_parameter_value, npar)

    # compute the order of the parameters
    parameter_order = zeros(Int64,npar)
    parameter_order[[1:pla_param-1; (pla_param+1):npar]] .= 1:npar-1
    parameter_order[pla_param] = npar

    function _loss_pla(θ, p)
      # construct the full parameter vector
      θ_full = [θ; fixed_parameter_value][parameter_order]
      loss(θ_full, p)
    end
end

# ─── Cell 15: PLA step function ───────────────────────────────────────────────
function pla_step(pla_param_index, loss, θ_init, lb, ub, p)
    npar = length(θ_init)
    lb_pla = [lb[1:pla_param_index-1]; lb[(pla_param_index+1):npar]]
    ub_pla = [ub[1:pla_param_index-1]; ub[(pla_param_index+1):npar]]
    initial_guess = [θ_init[1:pla_param_index-1]; θ_init[pla_param_index+1:end]]

    function _likelihood(pla_param)
        loss_function = loss_pla(loss, pla_param_index, pla_param, length(initial_guess)+1)
        optfunc = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
        optprob = OptimizationProblem(optfunc, initial_guess, p, lb = lb_pla, ub = ub_pla)

        optsol = solve(optprob, Optimization.LBFGS())
        optsol.objective, optsol.u
    end
end

# ─── Cell 16: PLA runner ──────────────────────────────────────────────────────
function run_pla(pla_param, param_range, param_optim, loss, initial_guess, lb_pla, ub_pla, p)

    # setup PLA likelihood
    step = pla_step(
        pla_param,
        loss,
        initial_guess,
        lb_pla, ub_pla, p
    )

    param_range = sort(unique([param_range; param_optim]))
    optim_index = findfirst(x -> x == param_optim, param_range)
    pla_likelihood_values = Float64[]
    other_parameter_values = typeof(initial_guess)[]
    parameter_values = typeof(param_optim)[]
    for px in param_range
        try
            l, _p = step(px)
            push!(pla_likelihood_values, l)
            push!(other_parameter_values, _p)
            push!(parameter_values, px)
        catch e
            throw(e)
            continue
        end
    end

    return PLAResult(pla_likelihood_values, parameter_values, optim_index, other_parameter_values)
end

# ─── Cell 17: Profile likelihood plotting function ────────────────────────────
function likelihood_profile!(ax, pla_result::PLAResult, param_index, param_name)
    parameter_values =  pla_result.parameter_values
    likelihood_values = pla_result.likelihood_values .- minimum(pla_result.likelihood_values)
    lines!(ax, parameter_values, likelihood_values, color=Makie.wong_colors()[1])
    #scatter!(ax, parameter_values, likelihood_values, color=Makie.wong_colors()[2])
end

# ─── Cell 18: Run PLA for 4-parameter EDES model ─────────────────────────────
# configure pla runs for each parameter
ranges = [
    range(1e-5, 0.05, length=800),
    range(0.001, 0.5, length=800),
    range(0.5, 6.0, length=400),
    range(3.0, 12.0, length=400)
]

pla_results = [
    run_pla(i, ranges[i], final[i], loss_known_sigma, final[1:4], [0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 10.0, 25.0], (prob, constants, data, final[5], final[6])) for i in eachindex(final[1:4])
];

# ─── Cell 19: Profile likelihood visualization for 4-parameter model ──────────
figure_likelihood = let f = Figure()
    positions = [(1,1), (1,2), (2,1), (2,2)]
    for (i, (pos, param_name)) in enumerate(zip(positions, ["k1", "k5", "k6", "k8"]))
        ax = Axis(getindex(f, pos...), xlabel=param_name, ylabel="ΔL", title="Profile Likelihood for $param_name")
        likelihood_profile!(ax, pla_results[i], i, param_name)
        scatter!(ax, [final[i]], [0], color=Makie.wong_colors()[2])
        limit_line = quantile(Chisq(1), 0.95)
        hlines!(ax, [limit_line], linestyle=:dash, color = Makie.wong_colors()[3])
        ylims!(ax, (0.0, 5.0))
    end

    f
end
save(joinpath(@__DIR__, "fig_cell19_pla_4param.png"), figure_likelihood)

# ─── Cell 20: 3-parameter model with k8 fixed to 7.25 ────────────────────────
#below is the solution for fixing the value of the k8 parameter to 7.25.
#This solution outlines the changes you need to make to the code to estimate 3 parameter rather than 4 (including specifing upper and lower bound for each estiamted parameter)

#firstly, we go to the construc_parameters function and fix k8 = 7.25
function construct_parameters(θ, c)

    # Estimated parameters
    k1 = θ[1]     # rate og glucose decay in stomach
    k5 = θ[2]     # rate of insulin dependent glucose uptake into peripheral tissues
    k6 = θ[3]     # rate of insulin secretion (proportional to Gb)
    k8 = 7.25     # rate of insulin secretion (derivative of Gpl) #to fix the value for k8 you can change the value here (an example is provided below)

    # Fixed parameters
    k2 = c[1]     # rate of glcuose transport from stomach to plasma
    k3 = c[2]     # rate of glucose effect on endogenous glucose production
    k4 = c[3]     # rate of insulin effect on endogenous glucose production
    k7 = c[4]     # rate of insulin secretion (integral of Gpl)
    k9 = c[5]     # delay parameter plasma to insterstitial insulin
    k10 = c[6]    # rate constant for degredation of insulin in remote compartment
    tau_i = c[7]  # time delay integrator function
    tau_d = c[8]  # time delay
    beta = c[9]   # conversion factor insulin
    Gren = c[10]  # threshold for renal excretion of glucose
    EGPb = c[11]  # basal rate of endogenous glucose production
    Km = c[12]    # Michealis Menten coeficient for glucose uptake into periphery
    f = c[13]     #
    Vg = c[14]    # volume of distribution for glucose (function of body weight)
    c1 = c[15]    # model constant
    sigma = c[16] # shape factor (appearance of meal)
    Dmeal = c[17] #amount of glucose in meal (mg)
    bw = c[18]    #body weight (kg)
    Gb = c[19]    #fasting glucose (mmol/l)
    Ib = c[20]    #fasting insulin (uIU/ml)
    return [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

#now generate the inital parameter guesses, in this case we provide only the upper and lower bounds for k1, k5, and k6.
initial_guess = generate_initial_guesses(1000, [0., 0., 0.,1e-4, 1e-4], [1., 0.5, 5.,100.0,100.0])


results = []
# now run the local optimisation for each initial guess
#
# note: here it is not unusual to get errors or warnings as for some combinations of parameters generatined by the generate_initial_guesses function can induce errors in the ODE solver and/or
# the optimisation algorithm by reaching a tolerance that induces a stop. By making use of the try-catch fundtion the loop with catch the error and continute to the next initial guess.
# This is also why we use many initial values, as we expect some parameter combinations to be unsuccessful. If you get an error for every parameter combination remove the try-catch and investigate
# if there is an error in your model code.
for guess in eachcol(initial_guess)
  try
    res = solve(OptimizationProblem(optf, vec(guess), (prob, constants, data), lb = [0,0,0,1e-4, 1e-4], ub=[1,0.5,10, 100, 100]), Optimization.LBFGS())
    push!(results, res)
  catch
    continue
  end
end;
println("3-param estimation: $(length(results))/1000 runs succeeded")
if isempty(results)
    error("All optimisation runs failed. Running first guess unguarded to diagnose:\n" *
          sprint(showerror, try
              solve(OptimizationProblem(optf, vec(initial_guess[:,1]), (prob, constants, data),
                    lb=[0,0,0,1e-4,1e-4], ub=[1,0.5,10,100,100]), Optimization.LBFGS())
              nothing
          catch e; e; end))
end
# find the parameterisation with the lowest cost/objective funtion value
best_index = argmin([r.objective for r in results])
final = results[best_index].u

#solve the ODE problem using the solve function - to get the model simulation for this parameter set
solution = solve(prob, p =construct_parameters(final, constants), u0=[0.0, data[2,1], data[3,1], 0.0]);

# visualize the solution against the data
solution_figure = let f = Figure(size=(500,500))

  ax_g_gut = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]", title="Gut Glucose")
  ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
  ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
  ax_i_int = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

  lines!(ax_g_gut, solution.t, solution[1,:], color=Makie.wong_colors()[1])
  lines!(ax_g_plasma, solution.t, solution[2,:], color=Makie.wong_colors()[1])
  lines!(ax_i_plasma, solution.t, solution[3,:], color=Makie.wong_colors()[1])
  lines!(ax_i_int, solution.t, solution[4,:], color=Makie.wong_colors()[1])

  band!(ax_g_plasma, solution.t, solution[2,:] .+ final[end-1], solution[2,:] .- final[end-1])
  band!(ax_i_plasma, solution.t, solution[3,:] .+ final[end], solution[3,:] .- final[end])

  scatter!(ax_i_plasma, data[1,:], data[3,:], color=Makie.wong_colors()[2])
  scatter!(ax_g_plasma, data[1,:], data[2,:], color=Makie.wong_colors()[2])

  f
end
save(joinpath(@__DIR__, "fig_cell20_fit_3param.png"), solution_figure)

# ─── Cell 21: PLA for 3-parameter model ──────────────────────────────────────
#configure pla runs for each parameter
ranges = [
    range(1e-5, 0.05, length=800),
    range(0.001, 0.5, length=800),
    range(0.5, 6.0, length=400),
]

pla_results = [
    run_pla(i, ranges[i], final[i], loss_known_sigma, final[1:3], [0.0, 0.0, 0.0], [1.0, 0.5, 10.0], (prob, constants, data, final[end-1], final[end])) for i in eachindex(final[1:3])
];

figure_likelihood = let f = Figure()
    positions = [(1,1), (1,2), (2,1)]
    for (i, (pos, param_name)) in enumerate(zip(positions, ["k1", "k5", "k6"]))
        ax = Axis(getindex(f, pos...), xlabel=param_name, ylabel="ΔL", title="Profile Likelihood for $param_name")
        likelihood_profile!(ax, pla_results[i], i, param_name)
        scatter!(ax, [final[i]], [0], color=Makie.wong_colors()[2])
        limit_line = quantile(Chisq(1), 0.95)
        hlines!(ax, [limit_line], linestyle=:dash, color = Makie.wong_colors()[3])
        ylims!(ax, (0.0, 5.0))
    end

    f
end
save(joinpath(@__DIR__, "fig_cell21_pla_3param.png"), figure_likelihood)

# ─── Cell 22: CGM data definition ────────────────────────────────────────────
# mean venous glucose and invsulin values measured during an OGTT
venous_glu = [5.4 7.1 8.6 8.9 8.5 7.6 6.6]
venous_ins = [7.9 36.7 64.8 75.6 79.6 80.5 68.7]
venous_time = [0 15 30 45 60 90 120]
venous_data = [venous_time;venous_glu;venous_ins]
#mean intersitial glucose measured using a CGM device during an OGTT
CGM_glu = [6.1 6.2 6.4 6.7 7.1 7.6 8.1 8.5 8.9 9.2 9.4 9.5 9.6 9.6 9.5 9.4 9.3 9.2 9.0 8.9 8.8 8.6 8.5 8.3 8.2 7.9]
CGM_time = [0	5	10	15	20	25	30	35	40	45	50	55	60	65	70	75	80	85	90	95	100	105	110	115	120	125]
CGM_ins = [7.9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] #the ODE solver needs fasting insulin to run, so I provide this value and zero for the remaining time point (will not be includedv in the error)
CGM_data = [CGM_time;CGM_glu;CGM_ins]

# ─── Cell 23: CGM loss function and parameter estimation ─────────────────────
#solution
#
#First we fit the EDES model to the CGM data. To do this we adjust the loss function to use only the glucose error
#define our loss function
#
#loss function takes in values for the estimated parameter and a tuple containing the ODE problem name, values for the fixed parameters and model constants, and the measured experimetnal data.
function loss_CGM(θ,(problem, constants ,data))
    #measured glucose and insulin data, including the sampling time points
      glucose_data = data[2,:]
      data_timepoints = data[1,:]

      #generate the full parameter vector by combining current values for the estimated parameters with the fixed values and constants.
      p = construct_parameters(θ, constants)

      # solve the ode problem for these parameter values, in this function we define the parameters as p
      # We also only save the ODE simulation at the timepoints that corrispond to the measured glucose and insulin data so we can calculate the error.
      # we also specify that of the four state varaibles [G_gut,  G_pl, I_pl, I_d] for calculating the error we are only interested in G_pl and I_pl
      # we also specify the initial values (U0) for each state variable ([G_gut,G_pl,I_pl,I_d])  for the ODE solver
      pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2], u0=[0.0, data[2,1], data[3,1], 0.0])
      sol = Array(pred)

      #calcualte the error between the measured glucose and insulin data and the EDES model simulation
      g_loss=(sol[1,:] - glucose_data)/maximum(glucose_data)

      # likelihood for the glucose data
      n = length(glucose_data)
      σ_g = θ[end-1]
      L_g = n*log(σ_g * sqrt(2π)) + 1/(2σ_g^2) * sum(abs2, g_loss)

      # weighted sum of the likelihoods
      return 0.5 * L_g
end
#we then run the parameter estimation with the new loss function for the three parameter EDES model (k1,k5,k6)
#we first define a new optfunction that directs the optimisation to our new loss function
optf = OptimizationFunction(loss_CGM, AutoForwardDiff())
    results = []
    for guess in eachcol(initial_guess)
      try
        res = solve(OptimizationProblem(optf, vec(guess), (prob, constants, CGM_data), lb = [0,0,0,1e-4, 1e-4], ub=[1,0.5,10, 100, 100]), Optimization.LBFGS())
        push!(results, res)
      catch
        continue
      end
    end;
    println("CGM estimation: $(length(results))/1000 runs succeeded")
    if isempty(results)
        error("All optimisation runs failed. Running first guess unguarded to diagnose:\n" *
              sprint(showerror, try
                  solve(OptimizationProblem(optf, vec(initial_guess[:,1]), (prob, constants, CGM_data),
                        lb=[0,0,0,1e-4,1e-4], ub=[1,0.5,10,100,100]), Optimization.LBFGS())
                  nothing
              catch e; e; end))
    end
    # find the parameterisation with the lowest cost/objective funtion value
    best_index = argmin([r.objective for r in results])
    final = results[best_index].u

    #solve the ODE problem using the solve function - to get the model simulation for this parameter set
    solution = solve(prob, p =construct_parameters(final, constants), u0=[0.0, CGM_data[2,1], 7.9, 0.0]);

    # visualize the solution against the data
    solution_figure = let f = Figure(size=(500,500))

      ax_g_gut = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]", title="Gut Glucose")
      ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
      ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
      ax_i_int = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

      lines!(ax_g_gut, solution.t, solution[1,:], color=Makie.wong_colors()[1])
      lines!(ax_g_plasma, solution.t, solution[2,:], color=Makie.wong_colors()[1])
      lines!(ax_i_plasma, solution.t, solution[3,:], color=Makie.wong_colors()[1])
      lines!(ax_i_int, solution.t, solution[4,:], color=Makie.wong_colors()[1])

      band!(ax_g_plasma, solution.t, solution[2,:] .+ final[end-1], solution[2,:] .- final[end-1])
      band!(ax_i_plasma, solution.t, solution[3,:] .+ final[end], solution[3,:] .- final[end])

      scatter!(ax_g_plasma, CGM_data[1,:], CGM_data[2,:], color=Makie.wong_colors()[2])

      f
    end
save(joinpath(@__DIR__, "fig_cell23_cgm_fit.png"), solution_figure)

# ─── Cell 24: Display final parameters ───────────────────────────────────────
println("Final estimated parameters (CGM fit): ", final)

# ─── Cell 25: PLA for CGM-only data ──────────────────────────────────────────
#The we run PLA for the three parameter EDES model fit to CGM glucose data only.
#
#Again, the first thing we do is modify the PLA loss function to use only CGM glucose (no insulin)
function loss_known_sigma_CGM(θ,(problem, constants ,data, sigma_g, sigma_i))
    #measured glucose and insulin data, including the sampling time points
      glucose_data = data[2,:]
      insulin_data = data[3,:]
      data_timepoints = data[1,:]

      #generate the full parameter vector by combining current values for the estimated parameters with the fixed values and constants.
      p = construct_parameters(θ, constants)

      # solve the ode problem for these parameter values
      pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2], u0=[0.0, data[2,1], data[3,1], 0.0])
      sol = Array(pred)

      #calcualte the error between the measured glucose and insulin data and the EDES model simulation
      g_loss=(sol[1,:] - glucose_data)/maximum(glucose_data)


      # likelihood for the glucose data
      n = length(glucose_data)
      σ_g = sigma_g
      L_g = n*log(σ_g * sqrt(2π)) + 1/(2σ_g^2) * sum(abs2, g_loss)


      # weighted sum of the likelihoods
      return 0.5 * L_g
end


#configure pla runs for each parameter
ranges = [
    range(1e-5, 0.02, length=800),
    range(0.001, 0.2, length=800),
    range(0.5, 20.0, length=400),
]

pla_results = [
    run_pla(i, ranges[i], final[i], loss_known_sigma_CGM, final[1:3], [0.0, 0.0, 0.0], [1.0, 0.5, 10.0], (prob, constants, CGM_data, final[end-1], final[end])) for i in eachindex(final[1:3])
];

figure_likelihood = let f = Figure()
    positions = [(1,1), (1,2), (2,1)]
    for (i, (pos, param_name)) in enumerate(zip(positions, ["k1", "k5", "k6"]))
        ax = Axis(getindex(f, pos...), xlabel=param_name, ylabel="ΔL", title="Profile Likelihood for $param_name")
        likelihood_profile!(ax, pla_results[i], i, param_name)
        scatter!(ax, [final[i]], [0], color=Makie.wong_colors()[2])
        limit_line = quantile(Chisq(1), 0.95)
        hlines!(ax, [limit_line], linestyle=:dash, color = Makie.wong_colors()[3])
        ylims!(ax, (0.0, 5.0))
    end

    f
end
save(joinpath(@__DIR__, "fig_cell25_pla_cgm.png"), figure_likelihood)
