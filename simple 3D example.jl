using TrajectoryOptimization
using Altro
using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots
using Rotations

const TO = TrajectoryOptimization
const RD = RobotDynamics

include("models/SimpleVine3D.jl")
include("src/visualize.jl")

# Create the model
model = SimpleVine3D(2, d=100., m_b = .0001, J_b = [1.0,1.0,1.0], stiffness = 500., damping=100.)
n,m = size(model)

# Generate initial state
rotations = [RotX(.1) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# # Check pin constraints
# model.c!(model.c, q0)
# xs = q0[1:7:end]
# ys = q0[2:7:end]
# zs = q0[3:7:end]
# plot([0;ys],[0;zs],aspect_ratio=:equal)

# dt = .005
# Z = [q0 q0]
# Z_meters = change_units_Z(model, Z)
# visualize!(model, Z_meters, dt)

# # Rollout dynamics
N = 2
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = ones(model.m, N-1)
dt = .005

for k = 2:N
    global Z
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
    println(maximum(abs.(model.c)))
end

# # visualize
# Z_meters = change_units_Z(model, Z)
# # visualize!(model, Z_meters, dt)

# # plot angles and velocity
# plot(Z[3:3:model.nq,:]')
