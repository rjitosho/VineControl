############################################################################################
#                                  IMPLICIT METHODS 								       #
############################################################################################
mutable struct MaxCoordsIntegrator <: MaxImp
    max_iters::Int
    line_iters::Int
    ϵ::T
end

function line_step!(x⁺_new, λ_new, x⁺, λ, Δs)
    
end

function integrate(::Type{MaxImp}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
    # initial guess
    λ = zeros(model.nc)
    x⁺ = copy(x)

    x⁺_new, λ_new = copy(x⁺), copy(λ)

    max_iters, line_iters, ϵ = model.mci.max_iters, model.mci.line_iters, model.mci.ϵ
    for i=1:max_iters  
        # Newton step    
        err = norm(fc(model, x⁺, x, u, λ, dt))
        F = fc_jacobian(model, x⁺, x, u, λ)
        Δs = F\err
       
        # line search
        j=0
        err_new = err + 1        
        while (err_new > err) && (j < line_iters)
            line_step!(x⁺_new, λ_new, x⁺, λ, Δs)
            err_new = norm(fc(model, x⁺_new, x, u, λ_new, dt))
            Δs /= 2
            j += 1
        end
        x⁺ .= x⁺_new
        λ .= λ_new

        # convergence check
        if err_new < ϵ
            return x⁺, λ
        end
    end

    throw("Newton did not converge. ")
end
