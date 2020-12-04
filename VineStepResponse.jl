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
model = SimpleVine3D(4, d=50., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 5000., damping=20000.)
n,m = size(model)

# Generate initial state
rotations = [RotX(0.) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = zeros(model.nv)
x0 = [q0; v0]

# Rollout dynamics
N = 20000
Z = zeros(model.n, N)
Z[:,1] = [q0;v0]
U = zeros(model.m, N-1)
U[1,:] .= 5000
Lam = zeros(model.nc, N-1)
dt = .001

for k = 2:N
    global Z
    print("k = $k\t")
    Z[:,k] = discrete_dynamics(PassThrough, model, SVector{model.n}(Z[:,k-1]),  SVector{model.m}(U[:,k-1]), 0, dt)
    Lam[:,k-1] = model.λ
end

# visualize
Z_meters = change_units_Z(model, Z)
# visualize!(model, Z_meters, dt)

using JLD2
# @save joinpath(@__DIR__, "step_response.jld2") Z U Lam N dt Z_meters
@load joinpath(@__DIR__, "step_response.jld2") Z U Lam N dt Z_meters

# plot y
plot(Z[2:7:model.nq,:]')

# --------------------------
include(joinpath(@__DIR__,"src/ilqr.jl"))

function c!(x) 
    c = copy(model.c)
    model.c!(c,x)
    return c
end

function f(x,u,dt)     
    x⁺ = discrete_dynamics(PassThrough, model, SVector{model.n}(x),  SVector{model.m}(u), 0, dt)
    λ = copy(model.λ)
    return x⁺, λ 
end

function getABCG(x⁺,x,u,λ,dt)
    discrete_jacobian(model, x⁺, [x;u;λ], dt)
end

function state_error(x,x0)
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

Q = 1.0*Matrix(I,2*model.nv,2*model.nv)
R = 1.0*Matrix(I,model.m,model.m)

xf = Z[:,end]
timesteps = 300
X = repeat(xf,outer=(1,timesteps+1))
Lam = repeat(Lam[:,end],outer=(1,timesteps))
U = repeat(U[:,end], outer=(1,timesteps))
K, l = backwardpass(X,Lam,U,getABCG,Q,R,Q,xf)
K6 = [K[1,6,i] for i=1:timesteps]
K3 = [K[1,3,i] for i=1:timesteps]
plot([K3 K6])

# Ku = K[:,:,1]
# x1, _ = f(xf,[1.],dt)
# X, Lam, U=stable_rollout(Ku,x1,U[:,1],f,dt,tf)
# plot(X[3,:])
# plot!(X[6,:])
