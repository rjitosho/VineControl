# import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots
using Rotations

const RD = RobotDynamics

include("models/SimpleVine3D.jl")

# Create the model
model = SimpleVine3D(2, d=100., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 50000., damping=30000.)
n,m = size(model)

# Generate initial state and control
using Random; Random.seed!(6)
rotations = [RotX(.2*rand()) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = .2*rand(model.nv)
x0 = [q0; v0]
u0 = 100*ones(model.m)
z0 = [x0; u0]
dt = .005

# compute next state
x⁺ = discrete_dynamics(PassThrough, model, SVector{model.n}(x0),  SVector{model.m}(u0), 0, dt)
l0 = copy(model.λ)

# compute jacobians
A,B,C,G = discrete_jacobian(model, x⁺, [x0;u0;l0], dt)

# compute deviations
function error_state_quat(model::SimpleVine3D, x, x0)
    nv, nq = model.nv, model.nq
    err = zeros(2*model.nv)
    dx = x-x0
    for i=1:model.nb
        err[6*(i-1) .+ (1:3)] = dx[7*(i-1) .+ (1:3)]
        dq = UnitQuaternion(x[7*(i-1) .+ (4:7)]) ⊖ UnitQuaternion(x0[7*(i-1) .+ (4:7)])
        err[6*(i-1) .+ (4:6)] = dq[:]
    end
    err[nv .+ (1:nv)] = dx[nq .+ (1:nv)]
    return err
end

# compute deviations
rotations = [rotations[i]+.01*randn() for i=1:model.nb]
q_new = generate_config(model, rotations)
x_new = [q_new; v0 + .001*randn(model.nv)]
# x_new = [q_new; v0]
# x_new = copy(x0)
dx = error_state_quat(model, x_new, x0)
# dx = x_new-x0
du = .001*rand(m)
x⁺_new = discrete_dynamics(PassThrough, model, SVector{model.n}(x_new),  SVector{model.m}(u0+du), 0, dt)
l_new = copy(model.λ)

# compare deviations 
dx⁺ = error_state_quat(model, x⁺_new, x⁺)
norm(dx⁺ - (A*dx + B*du + C*(l_new-l0)))
