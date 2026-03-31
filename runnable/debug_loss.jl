# Debug script for loss function
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "..")); Pkg.resolve(); Pkg.instantiate()
using OrdinaryDiffEq

# Copy the EDES ODE function
function edesode!(du, u, p, t)
    Ggut, Gpl, Ipl, Irem = u
    k1, k2v, k3v, k4v, k5, k6, k7v, k8, k9v, k10v, tau_iv, tau_dv, betav, Grenv, EGPbv, Kmv, fv, Vgv, c1v, sigmav, Dmealv, bwv, Gbv, Ibv = p

    du[1] = sigmav * k1^sigmav * t^(sigmav - 1) * exp(-(k1 * t)^sigmav) * Dmealv - k2v * Ggut

    gliv = EGPbv - k3v * (Gpl - Gbv) - k4v * betav * Irem
    ggut = k2v * (fv / (Vgv * bwv)) * Ggut
    u_ii = EGPbv * ((Kmv + Gbv) / Gbv) * (Gpl / (Kmv + Gpl))
    u_id = k5 * betav * Irem * (Gpl / (Kmv + Gpl))
    u_ren = c1v / (Vgv * bwv) * (Gpl - Grenv) * (Gpl > Grenv)
    du[2] = gliv + ggut - u_ii - u_id - u_ren

    i_pnc = betav^(-1) * (k6 * (Gpl - Gbv) + (k7v / tau_iv) * Gbv + k8 * tau_dv * du[2])
    i_liv = k7v * Gbv * Ipl / (betav * tau_iv * Ibv)
    i_int = k9v * (Ipl - Ibv)
    du[3] = i_pnc - i_liv - i_int
    du[4] = i_int - k10v * Irem
end

# Test data
data_time = [0 15 30 45 60 90 120]
data_glu = [5.4 7.1 8.6 8.9 8.5 7.6 6.6]
data_ins = [7.9 36.7 64.8 75.6 79.6 80.5 68.7]
data = [data_time; data_glu; data_ins]

# Parameters
bw = 70.0
Dmeal = 75.0e3
k2 = 0.28
k3 = 6.07e-3
k4 = 2.35e-4
k7 = 1.15
k9 = 3.83e-2
k10 = 2.84e-1
tau_i = 31.0
tau_d = 3.0
beta = 1.0
Gren = 9.0
EGPb = 0.043
Km = 13.2
f = 0.005551
c1 = 0.1
sigma = 1.4

Gb = data[2, 1]
Ib = data[3, 1]
Vg = 17.0 / bw

constants = [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]

# Create ODE problem
prob = ODEProblem(edesode!, [0.0, Gb, Ib, 0.0], (0.0, 240.0), constants)

# Test solve
println("Testing ODE solve...")
pred = solve(prob, Tsit5(), p=constants, saveat=data[1, :], save_idxs=[2, 3], u0=[0.0, data[2, 1], data[3, 1], 0.0])
println("ODE solve successful!")
println("Solution shape: $(size(Array(pred)))")
sol = Array(pred)
println("Solution: $(sol)")

# Test loss calculation
glucose_data = data[2, :]
insulin_data = data[3, :]
g_loss_raw = sol[1, :] - glucose_data
i_loss_raw = sol[2, :] - insulin_data
println("Raw g_loss: $(g_loss_raw)")
println("Raw i_loss: $(i_loss_raw)")

g_loss = g_loss_raw / maximum(glucose_data)
i_loss = i_loss_raw / maximum(insulin_data)
println("Normalized g_loss: $(g_loss)")
println("Normalized i_loss: $(i_loss)")

n = length(glucose_data)
sigma_g = 0.5
sigma_i = 0.3
L_g = n * log(sigma_g * sqrt(2π)) + 1 / (2 * sigma_g^2) * sum(abs2, g_loss)
L_i = n * log(sigma_i * sqrt(2π)) + 1 / (2 * sigma_i^2) * sum(abs2, i_loss)
println("L_g: $L_g")
println("L_i: $L_i")
println("Total loss: $(0.5 * L_g + 0.5 * L_i)")
