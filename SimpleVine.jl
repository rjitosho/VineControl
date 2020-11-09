mutable struct SimpleVine{T,S,D,V} <: AbstractModel
    θ0::T
    diam::T
    d::T
    k::S

    M::D
    MInv::D
    R::S
    Fext::V
    J::S
    c::V
    λ::V
    c!

    nb::Int
    nq::Int
    nc::Int

    n::Int
    m::Int
end

function SimpleVine(links;              # number of pin joints
                    d=5.0,              # distance from center of mass to body endpoint
                    stiffness=5000,     # pin joint stiffness
                    damping=30,         # pin joint damping
                    m_b=.001,           # mass
                    J_b=200,            # inertia
                    θ0 = 0.,            # initial heading
                    diam = 24.0,        # tube diameter in mm
                    )
    nb = links
    nq = 3*nb   # configuration dimension
    nc = 2*nb   # number of pin constraints

    n = 2*nq    # state size
    m = nb      # control size

    k = Matrix([Diagonal(stiffness*ones(links)) Diagonal(damping*ones(links))])
    M = Diagonal(SVector{3*nb}( repeat([m_b; m_b; J_b]; outer = [nb])))
    MInv = Diagonal(SVector{3*nb}( repeat([1/m_b; 1/m_b; 1/J_b]; outer = [nb])))
    Fext = zeros(nq)
    J = zeros(nc, nq)
    c = zeros(nc)
    λ = zeros(nc)

    R = zeros(3*nb, nb)
    R[3,1] = 1
    for i = 1:nb-1
        R[3*i,i+1] = -1
        R[3*i+3,i+1] = 1
    end

    # Pin joint contraint
    function c!(c,q)
        nc = length(c)

        # base pin
        c[1] = q[1] - d*cos(q[3])
        c[2] = q[2] - d*sin(q[3])

        # pin elements
        for i=2:Int(nc/2)
            x1 = q[3*i-5]
            y1 = q[3*i-4]
            Θ1 = q[3*i-3]
            x2 = q[3*i-2]
            y2 = q[3*i-1]
            Θ2 = q[3*i]
            c[2*i-1] = x2-x1-d*(cos(Θ1)+cos(Θ2))
            c[2*i] = y2-y1-d*(sin(Θ1)+sin(Θ2))
        end
    end

    return SimpleVine(θ0,diam,d,k,M,MInv,R,Fext,J,c,λ,c!,nb,nq,nc,n,m)
end

# Specify the state and control dimensions
RobotDynamics.state_dim(vine::SimpleVine) = vine.n
RobotDynamics.control_dim(vine::SimpleVine) = vine.m

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, m::SimpleVine, x::StaticVector, u::StaticVector, t, dt)
    nc = m.nc
    nq = m.nq

    q = view(x,1:nq)
    v = view(x,nq+1:2*nq)

    J = zeros(eltype(x), size(m.J))
    c = zeros(eltype(x), size(m.c))
    c! = m.c!

    MI = m.MInv

    # assemble vector of angles and angular velocities for calculating fExt
    θall = q[3:3:end] - [m.θ0; q[3:3:end-3]]
    θdall = v[3:3:end] - [0; v[3:3:end-3]]

    # torques
    F = -m.R * (m.k * [θall; θdall])
    F[3:3:end] += u

    # joint constraints
    c!(c,q)
    ForwardDiff.jacobian!(J, c!, ones(eltype(x), nc), q)

    # dynamics
    λ = -(J*MI*J')\(J*v + J*MI*F*dt + c/dt)
    v⁺ = v + MI*(J'*λ + F*dt)
    q⁺ = q + v⁺*dt
    return [q⁺; v⁺]
end

function generate_config(model, angles)
    # compute pin joint locations
    x = zeros(1)
    y = zeros(1)
    for i = 1:model.nb
        θ = angles[i]
        x = [x; x[end] + 2 * model.d * cos(θ)]
        y = [y; y[end] + 2 * model.d * sin(θ)]
    end

    # compute center of mass and orientation
    θ = atan.(y[2:end] - y[1:end-1], x[2:end] - x[1:end-1])
    x = (x[1:end-1] + x[2:end])/2
    y = (y[1:end-1] + y[2:end])/2

    return vec([x y θ]')
end