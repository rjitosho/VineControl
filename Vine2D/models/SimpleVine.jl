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

    # Mass matrix
    M = Diagonal(SVector{3*nb}( repeat([m_b; m_b; J_b]; outer = [nb])))

    # Map pin joint torque to maximal coordinates
    R = zeros(3*nb, nb)
    R[3,1] = 1
    for i = 1:nb-1
        R[3*i,i+1] = -1
        R[3*i+3,i+1] = 1
    end

    # Stiffness and damping matrices
    K = Diagonal(stiffness*ones(links)) 
    C = Diagonal(damping*ones(links))

    # Map configuration to pin joint angles
    D = zeros(nb, nq)
    [D[i,3*i]=1 for i=1:nb]
    [D[i+1,3*i]=-1 for i=1:nb-1]

    # Map body torque to maximal coordinates
    B = zeros(nq, nb)
    [B[3*i,i]=1 for i=1:nb]

    # Allocate matrices for dynamics
    Fext = zeros(nq)
    J = zeros(nc, nq)
    c = zeros(nc)
    λ = zeros(nc)

    # Pin joint contraint
    function c!(c,q)
        nc = length(c)

        # Base pin
        c[1] = q[1] - d*cos(q[3])
        c[2] = q[2] - d*sin(q[3])

        # Pin elements
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

# Compute maximal coordinate configuration given joint angles
function generate_config(model, angles)
    # Compute pin joint locations
    x = zeros(1)
    y = zeros(1)
    for i = 1:model.nb
        θ = angles[i]
        x = [x; x[end] + 2 * model.d * cos(θ)]
        y = [y; y[end] + 2 * model.d * sin(θ)]
    end

    # Compute center of mass and orientation
    θ = atan.(y[2:end] - y[1:end-1], x[2:end] - x[1:end-1])
    x = (x[1:end-1] + x[2:end])/2
    y = (y[1:end-1] + y[2:end])/2

    return vec([x y θ]')
end

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::SimpleVine, x::StaticVector, u::StaticVector, t, dt)
    # Unpack and allocate
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

    F = model.Fext
    J = model.J
    c = model.c
    λ = model.λ

    c! = model.c!

    # If called by ForwardDiff
    if !(eltype(x) <: Float64)
        F = zeros(eltype(x), size(model.Fext))
        J = zeros(eltype(x), size(model.J))
        c = zeros(eltype(x), size(model.c))
        λ = zeros(eltype(x), size(model.λ))
    end

    # Joint angles and angular velocities
    θall = D*q 
    θall[1] -= θ0
    θdall = D*v 

    # External impulse
    F .= (-R*(K*θall + C*θdall) + B*u)*dt

    # Set initial guess to q_k, v_k
    v⁺ = copy(v)
    q⁺ = copy(q)

    # Solve for next state
    max_iters = 10
    for i=1:max_iters
        # Joint constraints
        c!(c,q⁺)
        ForwardDiff.jacobian!(J, c!, ones(eltype(x), nc), q⁺)
        
        # Check break condition
        f = M*(v⁺-v) - J'*λ - F
        if norm([f;c]) < 1e-12
            println("breaking at iter: $i")
            break
        end
        i == max_iters && @warn "Max iters reached"

        # Newton solve
        A = [M -J';
            -J zeros(nc, nc)]
        b = [M*v + F; 
            (c + J*(q-q⁺))/dt]
        sol = A\b

        # Update        
        v⁺ = sol[1:nq]
        λ .= sol[nq+1:end]
        q⁺ = q + v⁺*dt
    end

    return [q⁺; v⁺]
end

function discrete_jacobian(model::SimpleVine, J, λ, x_next, z, dt)
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
    c! = model.c!
    
    # Partial (J'*λ) / partial q_k+1
    function JTλ(q)
        J_tmp = ForwardDiff.jacobian(c!, ones(eltype(q), nc), q)
        return J_tmp'*λ
    end
    J_q = ForwardDiff.jacobian(JTλ, x_next[1:nq])

    # Partial f / partial z_k
    df_dq = R*K*D*dt - J_q
    df_dv = R*C*D*dt - M
    df_du = -B*dt

    # Partial g / partial z_k
    dg_dz = [df_dq df_dv df_du;
            J zeros(nc,nq) zeros(nc, m)]

    # Partial g / partial [v_k+1; λ]
    dg_dv⁺ = [M-J_q*dt -J';
            J*dt zeros(nc,nc)]

    # Partial [v_k+1; λ] / partial z_k
    dv⁺_dq = -dg_dv⁺\dg_dz

    # Partial q_k+1 / partial z_k
    dq⁺_dq = dt*dv⁺_dq[1:nq,:]
    dq⁺_dq[1:nq,1:nq] += I

    # Return partial [q_k+1; v_k+1; λ] / partial z_k
    return [dq⁺_dq; dv⁺_dq]
end
