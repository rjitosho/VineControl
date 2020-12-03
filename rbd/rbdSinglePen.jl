import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using RigidBodyDynamics
using LinearAlgebra
using StaticArrays
using Plots

g = -9.81 # gravitational acceleration in z-direction
world = RigidBody{Float64}("world")
doublependulum = Mechanism(world; gravity = SVector(0, 0, g))

axis = SVector(0., 1., 0.) # joint axis
I_1 = .3341666666666666 # moment of inertia about joint axis
c_1 = -0.5 # center of mass location with respect to joint axis
m_1 = 1. # mass
frame1 = CartesianFrame3D("upper_link") # the reference frame in which the spatial inertia will be expressed
inertia1 = SpatialInertia(frame1,
    moment=I_1*Matrix(I,3,3),
    com=SVector(0, 0, c_1),
    mass=m_1)

upperlink = RigidBody(inertia1)
shoulder = Joint("shoulder", Revolute(axis))
before_shoulder_to_world = one(Transform3D,
    frame_before(shoulder), default_frame(world))
attach!(doublependulum, world, upperlink, shoulder,
    joint_pose = before_shoulder_to_world)

state = MechanismState(doublependulum)
set_configuration!(state, shoulder, pi/2)
set_velocity!(state, shoulder, 0.)

setdirty!(state)
ts, qs, vs = simulate(state, 4., Δt = .01);

plot([qs[i][1] for i=1:length(qs)])