# src/model.jl
# ODE model: build_constants, edesode!, construct_parameters, ode_solver

function build_constants(cfg::Dict)
    Vg = 17.0 / bw
    return [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, cfg["Gb"], cfg["Ib"]]
end

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

function construct_parameters(theta::Vector{Float64}, constants::Vector{Float64}, cfg::Dict)
    if cfg["k8_mode"] == :fixed
        k1v = theta[1]
        k5v = theta[2]
        k6v = theta[3]
        k8v = cfg["k8_fixed"]
    else
        k1v = theta[1]
        k5v = theta[2]
        k6v = theta[3]
        k8v = theta[4]
    end

    k2v = constants[1]; k3v = constants[2]; k4v = constants[3]; k7v = constants[4]
    k9v = constants[5]; k10v = constants[6]; tau_iv = constants[7]; tau_dv = constants[8]
    betav = constants[9]; Grenv = constants[10]; EGPbv = constants[11]; Kmv = constants[12]
    fv = constants[13]; Vgv = constants[14]; c1v = constants[15]; sigmav = constants[16]
    Dmealv = constants[17]; bwv = constants[18]; Gbv = constants[19]; Ibv = constants[20]

    return [k1v, k2v, k3v, k4v, k5v, k6v, k7v, k8v, k9v, k10v, tau_iv, tau_dv, betav, Grenv, EGPbv, Kmv, fv, Vgv, c1v, sigmav, Dmealv, bwv, Gbv, Ibv]
end

ode_solver = Tsit5()
