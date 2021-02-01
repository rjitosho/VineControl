using Pkg; Pkg.activate("rbd")
using RigidBodyDynamics
using LinearAlgebra
using StaticArrays
using Plots
using MeshCat
using MeshCatMechanisms

g = -9.81 # gravitational acceleration in z-direction
world = RigidBody{Float64}("world")
doublependulum = Mechanism(world; gravity = SVector(0, 0, g))

axis = SVector(0., 1., 0.) # joint axis
I_1 = 1/3 # moment of inertia about joint axis
I_1 = .292
c_1 = -0.5 # center of mass location with respect to joint axis
m_1 = 1. # mass
frame1 = CartesianFrame3D("upper_link") # the reference frame in which the spatial inertia will be expressed
inertia1 = SpatialInertia(frame1,
    moment=Matrix(Diagonal([I_1,I_1,.5*m_1*.1^2])),
    com=SVector(0, 0, c_1),
    mass=m_1)

upperlink = RigidBody(inertia1)
shoulder = Joint("shoulder", Revolute(axis))
before_shoulder_to_world = one(Transform3D,
    frame_before(shoulder), default_frame(world))
attach!(doublependulum, world, upperlink, shoulder,
    joint_pose = before_shoulder_to_world)

l_1 = -1. # length of the upper link
I_2 = 1/3 # moment of inertia about joint axis
I_2 = .292
c_2 = -0.5 # center of mass location with respect to joint axis
m_2 = 1. # mass
inertia2 = SpatialInertia(CartesianFrame3D("lower_link"),
    moment=Matrix(Diagonal([I_2,I_2,.5*m_2*.1^2])),
    com=SVector(0, 0, c_2),
    mass=m_2)
lowerlink = RigidBody(inertia2)
elbow = Joint("elbow", Revolute(axis))
before_elbow_to_after_shoulder = Transform3D(
    frame_before(elbow), frame_after(shoulder), SVector(0, 0, l_1))
attach!(doublependulum, upperlink, lowerlink, elbow,
    joint_pose = before_elbow_to_after_shoulder)

# mvis = MechanismVisualizer(doublependulum)
# set_configuration!(mvis, [0.0, 0.0])
# open(mvis)

state = MechanismState(doublependulum)
set_configuration!(state, shoulder, 0.3)
set_configuration!(state, elbow, 0.4)
set_velocity!(state, shoulder, 0.)
set_velocity!(state, elbow, 0.);

setdirty!(state)
ts, qs, vs = simulate(state, 1., Δt = 1e-3);

# animation = MeshCatMechanisms.Animation(mvis, ts, qs)
# setanimation!(mvis, animation)

pyplot()
N = length(qs)
plot([qs[i][1] for i=1:N])
plot!([qs[i][2] for i=1:N])