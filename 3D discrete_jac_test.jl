using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots
using Rotations

const RD = RobotDynamics

include("models/SimpleVine3D.jl")

# Create the model
model = SimpleVine3D(2, d=100., m_b = .01, J_b = [1.0,1.0,1.0], stiffness = 50000., damping=30000.)
n,m = size(model)

# Generate initial state and control
using Random; Random.seed!(6)
rotations = [RotX(.2*rand()) for i=1:model.nb]
q0 = generate_config(model, rotations)
v0 = .2*rand(model.nv)
x0 = [q0; v0]
u0 = 100*ones(model.m)
z0 = [x0; u0]

# Compute jacobian with ForwardDiff
dt = .005
function dyn(z)
    nb, nq, nv = model.nb, model.nq, model.nv
    n,m = size(model)
    M = model.M

    # Unpack
    q⁺ = z[1:nq]
    v⁺ = z[nq .+ (1:nv)]
    q = z[n .+ (1:nq)]
    v = z[n+nq .+ (1:nv)]
    u = z[2*n .+ (1:m)]
    λ = z[2*n+m+1:end]
    
    F = wrenches(model, [q⁺;v⁺], u) * dt

    J = zeros(eltype(z),size(model.J))
    J!(J,model.c!,q⁺,eltype(z))

    q_next = zeros(eltype(z),size(q⁺))
    q_next!(q_next,v⁺,q,dt)

    return [M*(v⁺-v) - J'*λ - F; q⁺ - q_next]
end

# Compute w ForwardDiff
x_next = discrete_dynamics(PassThrough, model, SVector{model.n}(x0),  SVector{model.m}(u0), 0, dt)
J_fd = ForwardDiff.jacobian(dyn, [x_next; z0; model.λ])

# Compute manually
# A,B,C,G = discrete_jacobian(model, model.J, model.λ, x_next, z0, dt)

# Check equal
# maximum(abs.(J_fd - [A B C]))
