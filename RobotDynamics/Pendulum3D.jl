using RobotDynamics
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools

# Define the model struct to inherit from `RigidBody{R}`
struct Pendulum3D{R,T} <: RigidBody{R}
    M::Array{T} # mass matrix
    b::T # damping
end
RobotDynamics.control_dim(::Pendulum3D) = 1

# Define some simple "getter" methods that are required to evaluate the dynamics
RobotDynamics.mass(model::Pendulum3D) = model.M

# Build model
T = Float64
R = UnitQuaternion{T}
M = 1.0*Matrix(I,6,6)
b = .1
model = Pendulum3D{R,T}(M, b)

function max_constraints(model, x)
    return [x[1:3] - UnitQuaternion(x[4:7]...) * [0;0;-.5];x[6:7]]
end

function max_constraints_jacobian(x⁺)
    J_big = ForwardDiff.jacobian(c!, x⁺)
    R⁺ = UnitQuaternion(x⁺[4:7]...)
    att_jac⁺ = Rotations.∇differential(R⁺)
    return [J_big[:,1:3] J_big[:,4:7]*att_jac⁺]
end

function wrenches(model, x⁺, x, u)
    [0;0;-9.81; u[1]-.1*x⁺[11];0;0] 
end

function fc(model, x⁺, x, u, λ, dt)
    c = max_constraints(model, x⁺)
    J = max_constraints_jacobian(x⁺)
    F = wrenches(model, x⁺, x, u) * dt

    v⁺ = x⁺[8:13]
    v = x[8:13]

    [model.M*(v⁺-v) - J'*λ - F; c]
end

function fc_jacobian(model, x⁺, x, u, λ, dt)
    nv, nc = 6, length(λ)
    function fc_aug(s)
        r⁺ = x[1:3] + s[1:3]*dt
        q⁺ = Rotations.params(Rotations.expm(s[4:6]*dt) * UnitQuaternion(x[4:7]...))
        fc(model, [r⁺;q⁺;s[1:6]], x, u, s[6 .+ (1:nc)], dt)
    end
    ForwardDiff.jacobian(fc_aug, [x⁺[8:end];λ])
end

function line_step!(x⁺_new, λ_new, x⁺, λ, Δs, x)
    # update lambda and v
    Δλ = Δs[7:end]
    λ_new .= λ - Δλ

    Δv⁺ = Δs[1:6]
    x⁺_new[7 .+ (1:6)] .= x⁺[7 .+ (1:6)] - Δv⁺    

    # compute configuration from v⁺
    x⁺_new[1:3] = x[1:3] + x⁺_new[7 .+ (1:3)]*dt
    x⁺_new[4:7] = Rotations.params(Rotations.expm(x⁺_new[7 .+ (4:6)]*dt) * UnitQuaternion(x[4:7]...))
    return    
end

# G = zeros(state_dim(model), RobotDynamics.state_diff_size(model))
# x,u = rand(model)
# z = KnotPoint(x,u,0.01)
# RobotDynamics.state_diff_jacobian!(G, model, x)
