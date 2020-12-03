# import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots
using Rotations
using JLD2

const RD = RobotDynamics

include("SimpleVine3D copy.jl")
include("visualize.jl")

# Create the model
model = SimpleVine3D(2, d=.50, m_b = 1., J_b =[0.0841667, 0.0841667,0.001666666666666667], stiffness = 0*50000., damping=0*30000.)
n,m = size(model)

# Generate initial state
rotations = [RotX(.2) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# Rollout dynamics
N = 2001
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = zeros(model.m, N-1)
dt = .001

for k = 2:N
    global Z
    print("k = $k\t")
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
end

# visualize
visualize!(model, Z, dt)

@save joinpath(@__DIR__, "swing.jld2") Z
@load joinpath(@__DIR__, "swing.jld2") Z

# _,N=size(Z)
# th1 = [AngleAxis(Quat(Z[4:7,i]...)).theta for i=1:N]
# sign1 = [AngleAxis(Quat(Z[4:7,i]...)).axis_x for i=1:N]
# th1 = th1.*sign1
# th2 = [AngleAxis(Quat(Z[11:14,i]...)).theta for i=1:N]
# sign2 = [AngleAxis(Quat(Z[11:14,i]...)).axis_x for i=1:N]
# th2 = th2.*sign2

# plot(th1)
# plot(th2)
# plot(th2-th1)

plot(Z[[2,9],:]')
