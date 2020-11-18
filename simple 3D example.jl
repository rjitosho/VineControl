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
model = SimpleVine3D(2, d=100., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 50000., damping=100.)
n,m = size(model)

# Generate initial state
rotations = [RotX(.1) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# Rollout dynamics
N = 10
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = ones(model.m, N-1)
dt = .005

for k = 2:N
    global Z
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
    println(maximum(abs.(model.c)))
end

# visualize
Z_meters = change_units_Z(model, Z)
visualize!(model, Z_meters, dt)

# plot angles
angle1 = []
angle2 = []
for  k = 1:N
    push!(angle1, Rotations.rotation_angle(UnitQuaternion(Z[4:7,k])))
    push!(angle2, Rotations.rotation_angle(UnitQuaternion(Z[11:14,k])))
end
plot([angle1 angle2])

# plot y
plot([Z[2,:] Z[9,:]])
