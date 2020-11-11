mutable struct SimpleVine <: AbstractModel
    nb::Int
    nq::Int
    nc::Int

    n::Int
    m::Int
    
    θ0::Number
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

    M = Diagonal(SVector{3*nb}( repeat([m_b; m_b; J_b]; outer = [nb])))
    # MInv = Diagonal(SVector{3*nb}( repeat([1/m_b; 1/m_b; 1/J_b]; outer = [nb])))

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

    return SimpleVine(nb,nq,nc,n,m,θ0,diam,d,M,R,K,C,D,B,Fext,J,c,λ,c!)
end

# Specify the state and control dimensions
RobotDynamics.state_dim(vine::SimpleVine) = vine.n
RobotDynamics.control_dim(vine::SimpleVine) = vine.m

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::SimpleVine, x::StaticVector, u::StaticVector, t, dt)
    # unpack and allocate
    nc = model.nc
    nq = model.nq

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
    θall = D*q #q[3:3:end] - [model.θ0; q[3:3:end-3]]
    θall[1] -= θ0
    θdall = D*v #v[3:3:end] - [0; v[3:3:end-3]]

    # external impulse
    F = -R * (K*θall + C*θdall) + B * u 
    F *= dt # compute impulse

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
        q⁺ = q + v⁺*dt
    end

    # store values
    if eltype(x) <: Float64
        model.J .= J
        model.c .= c
        model.λ .= λ
    end
    return [q⁺; v⁺]
end

function discrete_jacobian(model::SimpleVine, x_next, z, dt)
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
    λ = model.λ
    c! = model.c!

    # partial J / partial q_k+1
    function JTλ(q)
        J_tmp = zeros(eltype(q), size(J))
        ForwardDiff.jacobian!(J_tmp, c!, ones(eltype(q), nc), q)
        return J_tmp'*λ
    end
    J_q = zeros(nq, nq)
    ForwardDiff.jacobian!(J_q, JTλ, x_next[1:nq])

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

# compute maximal coordinate configuration given joint angles
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