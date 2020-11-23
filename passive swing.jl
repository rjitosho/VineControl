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
model = SimpleVine3D(8, d=25., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 0*50000., damping=0*30000.)
n,m = size(model)

# Generate initial state
rotations = [RotX(.2) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# Rollout dynamics
N = 2
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = zeros(model.m, N-1)
dt = .005

for k = 2:N
    global Z
    println("k = $k")
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
    # println(maximum(abs.(model.c)))
end

# visualize
# Z_meters = change_units_Z(model, Z)
# visualize!(model, Z_meters, dt)

# plot y
plot([Z[2,:] Z[9,:]])

