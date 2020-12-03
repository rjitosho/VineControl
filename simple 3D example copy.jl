# import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# using TrajectoryOptimization
# using Altro
using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Rotations
using Plots

# const TO = TrajectoryOptimization
const RD = RobotDynamics

include("models/SimpleVine3D.jl")
include("src/visualize.jl")

# Create the model
model = SimpleVine3D(4, d=50., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 5000., damping=10000.)
n,m = size(model)

# Generate initial state
rotations = [RotX(.1) for i=1:model.nb]
rotations = [RotX(0.) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# Rollout dynamics
N = 1001
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = zeros(model.m, N-1)
U[1,1:150] = 5000*sin.(collect(1:150)/150*pi)
U[2,601:750] = 5000*sin.(collect(1:150)/150*pi)
Lam = zeros(model.nc, N-1)
dt = .005

for k = 2:N
    global Z
    print("k = $k\t")
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
    Lam[:,k-1] = model.λ
    # println(maximum(abs.(model.c)))
end

# visualize
Z_meters = change_units_Z(model, Z)
visualize!(model, Z_meters, dt)

# plot y
plot([Z[2,:] Z[9,:]])
plot!([Z[1,:] Z[8,:]])
