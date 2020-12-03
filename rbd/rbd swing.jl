import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using RigidBodyDynamics
using LinearAlgebra
using StaticArrays
using Rotations
using Plots
using JLD2

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
shoulder = Joint("shoulder", QuaternionSpherical{Float64}())
before_shoulder_to_world = one(Transform3D,
    frame_before(shoulder), default_frame(world))
attach!(doublependulum, world, upperlink, shoulder,
    joint_pose = before_shoulder_to_world)


# l_1 = -1.0 # length of the upper link
# I_2 = .3341666666666666 # moment of inertia about joint axis
# c_2 = -0.5 # center of mass location with respect to joint axis
# m_2 = 1. # mass
# inertia2 = SpatialInertia(CartesianFrame3D("lower_link"),
#     moment=I_2*Matrix(I,3,3),
#     com=SVector(0, 0, c_2),
#     mass=m_2)
# lowerlink = RigidBody(inertia2)
# elbow = Joint("elbow", QuaternionSpherical{Float64}())
# before_elbow_to_after_shoulder = Transform3D(
#     frame_before(elbow), frame_after(shoulder), SVector(0, 0, l_1))
# attach!(doublependulum, upperlink, lowerlink, elbow,
#     joint_pose = before_elbow_to_after_shoulder)

state = MechanismState(doublependulum)
set_configuration!(state, shoulder, RotX(.2))
# set_configuration!(state, elbow, RotX(0.))
set_velocity!(state, shoulder, zeros(3))
# set_velocity!(state, elbow, zeros(3));

setdirty!(state)
ts, qs, vs = simulate(state, 2., Δt = .01);
qs = [qs[i][:] for i=1:length(qs)]
vs = [vs[i][:] for i=1:length(vs)]

rbd_th1 = [AngleAxis(Quat(qs[i][1:4]...)).theta for i=1:length(qs)]
rbd_sign1 = [AngleAxis(Quat(qs[i][1:4]...)).axis_x for i=1:length(qs)]
rbd_th1 = rbd_th1 .* rbd_sign1
# rbd_th2 = [AngleAxis(Quat(qs[i][5:8]...)).theta for i=1:length(qs)]
# rbd_sign2 = [AngleAxis(Quat(qs[i][5:8]...)).axis_x for i=1:length(qs)]
# rbd_th2 = rbd_th2 .* rbd_sign2

@save joinpath(@__DIR__, "rbd_swing.jld2") ts rbd_th1
@load joinpath(@__DIR__, "rbd_swing.jld2") ts rbd_th1

plot(rbd_th1)
# plot(rbd_th2)

