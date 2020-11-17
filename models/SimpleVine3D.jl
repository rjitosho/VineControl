mutable struct SimpleVine3D <: AbstractModel
    nb::Int
    nq::Int
    nv::Int
    nc::Int

    n::Int
    m::Int
    
    θ0
    diam::Number
    d::Number
    M

    R
    K
    C
    D
    B 
    Fext

    J
    c
    λ
    c!
end

function SimpleVine3D(links;              # number of pin joints
                    d=5.0,              # distance from center of mass to body endpoint
                    stiffness=5000,     # pin joint stiffness
                    damping=30,         # pin joint damping
                    m_b=.001,           # mass
                    J_b=[200;200;200],            # inertia
                    θ0 = 0.,            # initial heading
                    diam = 24.0,        # tube diameter in mm
                    )
    nb = links
    nq = 7*nb   # configuration dimension
    nv = 6*nb   # velocity dimension
    nc = 3*nb   # number of pin constraints

    n = 13*nb    # state size
    m = nb       # control size

    # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    M = Diagonal(repeat([m_b; m_b; m_b; J_b]; outer = [nb]))

    R = zeros(3*nb, nb)
    R[3,1] = 1
    for i = 1:nb-1
        R[3*i,i+1] = -1
        R[3*i+3,i+1] = 1
    end

    K = Diagonal(stiffness*ones(links)) 
    C = Diagonal(damping*ones(links))

    D = zeros(nb, nq)
    [D[i,3*i]=1 for i=1:nb]
    [D[i+1,3*i]=-1 for i=1:nb-1]

    B = zeros(nq, nb)
    [B[3*i,i]=1 for i=1:nb]

    Fext = zeros(nq)

    J = zeros(nc, nq)
    c = zeros(nc)
    λ = zeros(nc)

    # Pin joint contraint
    function c!(c,q)
        nc = length(c)
        endpoint = [0;0;-model.d]

        # base pin
        c[1:3] = q[1:3] - UnitQuaternion(q[4:7]) * endpoint

        # pin elements
        for i=1:model.nb-1
            p1 = q[7*(i-1) .+ (1:3)]
            r1 = UnitQuaternion(q[7*(i-1) .+ (4:7)])
            p2 = q[7*i .+ (1:3)]
            r2 = UnitQuaternion(q[7*i .+ (4:7)])
            c[3*i .+ (1:3)] = (p2 - r2*endpoint) - (p1 + r1*endpoint)
        end
    end

    return SimpleVine3D(nb,nq,nv,nc,n,m,θ0,diam,d,M,R,K,C,D,B,Fext,J,c,λ,c!)
end

# Specify the state and control dimensions
RobotDynamics.state_dim(vine::SimpleVine3D) = vine.n
RobotDynamics.control_dim(vine::SimpleVine3D) = vine.m

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::SimpleVine3D, x::StaticVector, u::StaticVector, t, dt)
    # unpack and allocate
    nb = model.nb
    nq = model.nq
    nv = model.nv
    nc = model.nc

    M = Matrix(model.M)
    K = model.K
    C = model.C
    D = model.D
    B = model.B
    R = model.R
    θ0 = model.θ0

    q = x[1:nq]
    v = x[nq+1:end]

    J = zeros(eltype(x), size(model.J))
    c = zeros(eltype(x), size(model.c))
    c! = model.c!

    # joint angles and angular velocities
    # θall = D*q #q[3:3:end] - [model.θ0; q[3:3:end-3]]
    # θall[1] -= θ0
    # θdall = D*v #v[3:3:end] - [0; v[3:3:end-3]]

    # # external impulse
    # F = -R * (K*θall + C*θdall) + B * u 
    # F *= dt # compute impulse

    # gravity
    F = zeros(model.nv)
    F[3:6:end] .= -M[1,1]*9810

    # rotation    
    # ωdot = Jinv*(- ω × (J*ω))

    # set initial guess to q_k, v_k
    v⁺ = copy(v)
    q⁺ = copy(q)
    λ = zeros(eltype(x), size(model.λ))

    for i=1:3
        # joint constraints
        c!(c,q⁺)
        ForwardDiff.jacobian!(J, c!, ones(eltype(x), nc), q⁺)

        # Newton solve
        A = [M -J';
            -J zeros(nc, nc)]
        b = [M*v + F; 
            (c + J*(q-q⁺))/dt]

        # Unpack
        sol = A\b
        v⁺ = sol[1:nq]
        λ = sol[nq+1:end]
        for i=1:nb
            # position
            p_ind = 7*(i-1) .+ (1:3)
            q⁺[p_ind] = q[p_ind] + v⁺[6*(i-1) .+ (1:3)]*dt

            # orientation
            r_ind = 7*(i-1) .+ (4:7)
            r = UnitQuaternion(q[r_ind])
            q⁺[r_ind] = r + Rotations.kinematics(r, v⁺[6*(i-1) .+ (4:6)])*dt
        end
    end

    # store values
    if eltype(x) <: Float64
        model.J .= J
        model.c .= c
        model.λ .= λ
    end
    return [q⁺; v⁺]
end

function discrete_jacobian(model::SimpleVine3D, x_next, z, dt)
    # Unpack
    n, m = size(model)
    x = z[1:n]
    u = z[n+1:end]
  
    nb = model.nb
    nc = model.nc
    nq = model.nq

    R = model.R
    K = model.K
    C = model.C
    D = model.D
    B = model.B
    M = model.M
    
    J = model.J
    c = model.c
    λ = model.λ
    c! = model.c!

    # q⁺ = x_next[1:model.nq]
    # c!(c,q⁺)
    # ForwardDiff.jacobian!(J, c!, ones(nc), q⁺)

    # partial J / partial q_k+1
    function JTλ(q)
        J_tmp = zeros(eltype(q), size(J))
        ForwardDiff.jacobian!(J_tmp, c!, ones(eltype(q), nc), q)
        return J_tmp'*λ
    end
    J_q = zeros(nq, nq)
    # ForwardDiff.jacobian!(J_q, JTλ, x_next[1:nq])

    # partial g / partial v_k+1 and λ
    dg_dv⁺ = [M-J_q*dt -J';
            J*dt zeros(nc,nc)]

    # partial g / partial z_k
    df_dq = R*K*D*dt - J_q
    df_dv = R*C*D*dt - M
    df_du = -B*dt

    dg_dz = [df_dq df_dv df_du;
            J zeros(nc,nq) zeros(nc, m)]

    # partial v_k+1 / partial z
    dv⁺_dq = (-dg_dv⁺\dg_dz)[1:nq,:]

    # partial q_k+1 / partial z
    dq⁺_dq = dt*dv⁺_dq
    dq⁺_dq[1:nq,1:nq] += I

    # return partial x_k+1 / partial z_k
    return [dq⁺_dq; dv⁺_dq]
end

# compute maximal coordinate configuration given body rotations
function generate_config(model, rotations)
    @assert model.nb == length(rotations)
    pin = zeros(3)
    q = zeros(0)   
    d = [0,0,-model.d]
    for i = 1:model.nb
        r = UnitQuaternion(rotations[i])
        delta = r * d
        q = [q; pin+delta; Rotations.params(r)]
        pin += 2*delta
    end
    return q
end