using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots

const RD = RobotDynamics

include("models/SimpleVine.jl")
include("src/visualize.jl")

# Create the model
model = SimpleVine(2, d=100., m_b = .0001, J_b = 1.0, stiffness = 500., damping=100.)
n,m = size(model)

# Generate initial state
angles = zeros(model.nb)
q0 = generate_config(model, angles)
v0 = zeros(model.nq)
x0 = [q0; v0]

# Discrete jacobian
dt = .005
u0 = 100*ones(model.m)
z0 = [x0; u0]
x_next = dyn(z0)
J_manual = discrete_jacobian(model, x_next, z0, dt)

# Compute with ForwardDiff
function dyn(z)
    x = SVector{model.n}(z[1:model.n])
    u = SVector{model.m}(z[model.n+1:end])
    discrete_dynamics(PassThrough, model, x, u, 0, dt)
end
J_fd = zeros(n, n+m)
ForwardDiff.jacobian!(J_fd, dyn, z0)

# Check equal
maximum(abs.(J_fd - J_manual))
