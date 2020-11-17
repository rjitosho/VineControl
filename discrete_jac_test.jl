using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots

const RD = RobotDynamics

include("models/SimpleVine.jl")

# Create the model
model = SimpleVine(2, d=100., m_b = .0001, J_b = 1.0, stiffness = 500., damping=100.)
n,m = size(model)

# Generate initial state and control
using Random; Random.seed!(6)
angles = rand(model.nb)
q0 = generate_config(model, angles)
v0 = rand(model.nq)
x0 = [q0; v0]
u0 = 100*ones(model.m)
z0 = [x0; u0]

# Compute jacobian with ForwardDiff
dt = .005
function dyn(z)
    x = SVector{model.n}(z[1:model.n])
    u = SVector{model.m}(z[model.n+1:end])
    discrete_dynamics(PassThrough, model, x, u, 0, dt)
end
J_fd = ForwardDiff.jacobian(dyn, z0)

# Compute manually
x_next = dyn(z0)
J_manual = discrete_jacobian(model, model.J, model.λ, x_next, z0, dt)

# Check equal
maximum(abs.(J_fd - J_manual[1:n,:]))
