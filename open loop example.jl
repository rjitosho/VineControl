using TrajectoryOptimization
using Altro
using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots

const TO = TrajectoryOptimization
const RD = RobotDynamics

include("SimpleVine.jl")
include("visualize.jl")

# Create the model
model = SimpleVine(2, d=100., m_b = .0001, J_b = 1.0, stiffness = 500., damping=100.)
n,m = size(model)

# Generate initial state
angles = zeros(model.nb)
q0 = generate_config(model, angles)
v0 = zeros(model.nq)
x0 = [q0; v0]

# Rollout dynamics
N = 500
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = 100*ones(model.m, N-1)
dt = .005

for k = 2:N
    global Z
    print(".")
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
end

# visualize
Z_meters = change_units_Z(model, Z)
# visualize!(model, Z_meters)

# plot angles and velocity
plot(Z[3:3:end,:]')
