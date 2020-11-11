using TrajectoryOptimization
using Altro
using RobotDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Plots

const TO = TrajectoryOptimization
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
x0 = [q0;v0]

# Generate final state
angles = .1*ones(model.nb)
qf = generate_config(model, angles)
vf = zeros(model.nq)
xf = [qf;vf]

# Objective
x0 = SA[x0...]  # initial state
xf = SA[xf...]   # final state

N = 300
tf = .005*(N-1)
Q = Diagonal(@SVector ones(n))
R = .0001*Diagonal(@SVector ones(m))
Qf = Diagonal(@SVector ones(n))
obj = LQRObjective(Q, R, Qf, xf, N)

# Create and solve problem
prob = Problem(model, obj, xf, tf, x0=x0, integration=RD.PassThrough)
solver = ALTROSolver(prob, SolverOptions(verbose = 1))
set_options!(solver, show_summary=true)
solve!(solver);         # solve with ALTRO

# Get the state and control trajectories
# X = states(solver)
# U = controls(solver)

# X_mat = hcat(X...)
# U_mat = hcat(U...)

# plot(X_mat[[3,6], :]')
# plot(U_mat')

# visualize
# Z_meters = change_units_Z(model, X_mat)
# visualize!(model, Z_meters)