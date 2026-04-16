# src/loss.jl
# Loss function for parameter optimization

function loss_universal(theta::Vector{Float64}, payload)
    problem, constants, data, cfg = payload

    glucose_data = data[2, :]
    insulin_data = data[3, :]
    data_timepoints = data[1, :]

    # Check parameter bounds and return large penalty if violated
    for (i, val) in enumerate(theta)
        if val < cfg["lb"][i] || val > cfg["ub"][i]
            return 1e10  # Large penalty for out-of-bounds
        end
    end

    p = construct_parameters(theta, constants, cfg)
    local pred
    try
        pred = solve(problem, ode_solver, p = p, saveat = data_timepoints, save_idxs = cfg["save_idxs"],
                     u0 = [0.0, data[2, 1], data[3, 1], 0.0])
    catch
        return 1e10
    end

    if cfg["uses_insulin_data"]
        sol = Array(pred)
        size(sol, 2) != length(glucose_data) && return 1e10

        # Raw residuals (no normalization for likelihood)
        g_resid = sol[1, :] - glucose_data
        i_resid = sol[2, :] - insulin_data
        any(!isfinite, g_resid) && return 1e10
        any(!isfinite, i_resid) && return 1e10

        n = length(glucose_data)
        sigma_g = theta[end - 1]
        sigma_i = theta[end]

        # Proper negative log-likelihood for normal distribution
        L_g = n * log(sigma_g) + sum(abs2, g_resid) / (2 * sigma_g^2)
        L_i = n * log(sigma_i) + sum(abs2, i_resid) / (2 * sigma_i^2)
        (isfinite(L_g) && isfinite(L_i)) || return 1e10
        return L_g + L_i
    else
        sol = vec(Array(pred))
        length(sol) == length(glucose_data) || return 1e10

        # Raw residuals (no normalization for likelihood)
        g_resid = sol - glucose_data
        any(!isfinite, g_resid) && return 1e10

        n = length(glucose_data)
        sigma_g = theta[end - 1]

        # Proper negative log-likelihood for normal distribution
        L_g = n * log(sigma_g) + sum(abs2, g_resid) / (2 * sigma_g^2)
        isfinite(L_g) || return 1e10
        return L_g
    end
end
