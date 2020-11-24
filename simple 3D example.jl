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
model = SimpleVine3D(2, d=100., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 50000., damping=30000.)
n,m = size(model)

# Generate initial state
rotations = [RotX(.1) for i=1:model.nb]
rotations = [RotX(0.) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# Rollout dynamics
N = 401
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = zeros(model.m, N-1)
U[1,1:200] = 10000*sin.(collect(1:200)/200*pi)
U[2,101:300] = 10000*sin.(collect(1:200)/200*pi)
U[3,201:400] = 10000*sin.(collect(1:200)/200*pi)
dt = .005

for k = 2:N
    global Z
    println("k = $k")
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
    # println(maximum(abs.(model.c)))
end

# visualize
Z_meters = change_units_Z(model, Z)
# visualize!(model, Z_meters, dt)

# plot y
plot([Z[2,:] Z[9,:]])
plot!([Z[1,:] Z[8,:]])

# time = 1:138
# plot([Z[2,time] Z[9,time]])
# plot!([Z[1,time] Z[8,time]])
