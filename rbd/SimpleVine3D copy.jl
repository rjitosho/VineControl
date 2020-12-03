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

    W
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
    m = 3       # control size

    # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    M = Diagonal(repeat([m_b; m_b; m_b; J_b]; outer = [nb]))

    W = zeros(3*nb, nb)
    W[3,1] = 1
    for i = 1:nb-1
        W[3*i,i+1] = -1
        W[3*i+3,i+1] = 1
    end

    K = Diagonal(stiffness*ones(links)) 
    C = Diagonal(damping*ones(links))

    D = zeros(nb, nq)
    [D[i,3*i]=1 for i=1:nb]
    [D[i+1,3*i]=-1 for i=1:nb-1]

    B = [1      -1/2        -1/2;
         0 -sqrt(3)/2 sqrt(3)/2;
         0      0           0]

    Fext = zeros(nv)

    J = zeros(nc, nv)
    c = zeros(nc)
    λ = zeros(nc)

    # Pin joint contraint
    function c!(c,q)
        nc = length(c)
        endpoint = [0;0;-model.d]

        # base pin
        c[1:3] = q[1:3] - UnitQuaternion(q[4:7]...) * endpoint

        # pin elements
        for i=1:model.nb-1
            r1 = q[7*(i-1) .+ (1:3)]
            R1 = UnitQuaternion(q[7*(i-1) .+ (4:7)]...)
            r2 = q[7*i .+ (1:3)]
            R2 = UnitQuaternion(q[7*i .+ (4:7)]...)
            c[3*i .+ (1:3)] = (r2 - R2*endpoint) - (r1 + R1*endpoint)
        end
    end

    return SimpleVine3D(nb,nq,nv,nc,n,m,θ0,diam,d,M,W,K,C,D,B,Fext,J,c,λ,c!)
end

# Specify the state and control dimensions
RobotDynamics.state_dim(vine::SimpleVine3D) = vine.n
RobotDynamics.control_dim(vine::SimpleVine3D) = vine.m

function bend_error(R1, R2)
    endpoint = [0;0;-model.d]
    R1 = UnitQuaternion(R1...)
    R2 = UnitQuaternion(R2...)

    # Bending stiffness
    err = (R2 ⊖ R1).err
    bend_axis = cross(R1*endpoint, R2*endpoint)
    norm(bend_axis) > 1e-6 && (bend_axis /= norm(bend_axis))
    bend_err = dot(bend_axis, err) * bend_axis
    
    # Torsion stiffness
    twist_err = err - bend_err

    return bend_err, bend_axis, twist_err 
end

function wrenches(model::SimpleVine3D, x, u)
    nb, nq, nv = model.nb, model.nq, model.nv

    q = x[1:nq]
    v = x[1+nq:end]

    # Gravity
    F = zeros(eltype(x), nv)
    F[3:6:end] .= -model.M[1,1]*9.810

    # Rotation
    J = model.M[4:6,4:6]
    for i=1:nb
        ω_idx = 6*(i-1) .+ (4:6)
        ω = v[ω_idx]
        F[ω_idx] += model.B*u #- ω × (J*ω) 
    end

    # Base pin
    bend_err, bend_axis, twist_err = bend_error([1.0,0,0,0], q[4:7])
    # F[4:6] += -model.K[1,1]*bend_err
    # F[4:6] += -model.C[1,1]*dot(bend_axis, v[4:6])*bend_axis
    
    # Other pins
    for i=1:nb-1
        ω1_idx = 6*(i-1) .+ (4:6)
        ω2_idx = 6*i .+ (4:6)

        # Spring
        bend_err, bend_axis, twist_err = bend_error(q[7*(i-1) .+ (4:7)], q[7*i .+ (4:7)])
        # F[ω1_idx] += model.K[i+1,i+1]*bend_err + 5000*twist_err
        # F[ω2_idx] += -model.K[i+1,i+1]*bend_err - 5000*twist_err

        # Damping
        ω1 = v[ω1_idx]
        ω2 = v[ω1_idx]
        diff = dot(bend_axis, ω2-ω1)*bend_axis
        twist_diff = (ω2-ω1) - diff
        # F[ω1_idx] += model.C[i+1,i+1]*diff + 3000*twist_diff
        # F[ω2_idx] += -model.C[i+1,i+1]*diff - 3000*twist_diff
    end
    
    return F
end

function J!(J,c!,q⁺,type)
    nb = Int(length(q⁺)/7)
    nc, _ = size(J)
    J_big = ForwardDiff.jacobian(c!, ones(type, nc), q⁺)
    
    for i=1:nb
        # dc/dr
        J[:, 6*(i-1) .+ (1:3)] = J_big[:, 7*(i-1) .+ (1:3)]

        # dc/dq
        att_jac = Rotations.∇differential(UnitQuaternion(q⁺[7*(i-1) .+ (4:7)]...))
        J[:, 6*(i-1) .+ (4:6)] = J_big[:, 7*(i-1) .+ (4:7)] * att_jac
    end
end

function q_next!(q⁺,v⁺,q,dt)
    nb = Int(length(q⁺)/7)
    for i=1:nb
        # position
        r_idx = 7*(i-1) .+ (1:3)
        q⁺[r_idx] = q[r_idx] + v⁺[6*(i-1) .+ (1:3)]*dt

        # orientation
        R_idx = 7*(i-1) .+ (4:7)
        R = UnitQuaternion(q[R_idx]...)
        ω⁺ = v⁺[6*(i-1) .+ (4:6)]
        R⁺ = Rotations.params(Rotations.expm(ω⁺*dt) * R)
        q⁺[R_idx] = R⁺/norm(R⁺)
    end
end

function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::SimpleVine3D, 
                                        x::StaticVector, u::StaticVector, t, dt)
    # unpack and allocate
    nb, nq, nv, nc = model.nb, model.nq, model.nv, model.nc
    M = Matrix(model.M)

    q = x[1:nq]
    v = x[nq+1:end]

    F = model.Fext
    J = model.J
    c = model.c
    λ = model.λ

    c! = model.c!

    # If called by ForwardDiff
    if !(eltype(x) <: Float64)
        # F = zeros(eltype(x), size(model.Fext))
        # J = zeros(eltype(x), size(model.J))
        # c = zeros(eltype(x), size(model.c))
        # λ = zeros(eltype(x), size(model.λ))
        F = convert(typeof(x), model.Fext)
        J = convert(typeof(x), model.J)
        c = convert(typeof(x), model.c)
        λ = convert(typeof(x), model.λ)
    end

    # set initial values
    v⁺ = copy(v)
    q⁺ = copy(q)
    F .= wrenches(model, x, u) * dt
    J!(J,c!,q⁺,eltype(x))
    c!(c,q⁺)
    
    max_iters = 100
    for i=1:max_iters        
        # dq_dv
        dq_dv = Matrix(dt*I, nv, nv)
        for i=1:nb            
            R = UnitQuaternion(q[7*(i-1) .+ (4:7)]...)
            R⁺ = UnitQuaternion(q⁺[7*(i-1) .+ (4:7)]...)
            att_jac = Rotations.∇differential(R⁺)

            ω_idx = 6*(i-1) .+ (4:6)
            f(ω⁺) = Rotations.params(Rotations.expm(ω⁺*dt) * R)
            dq_dv[ω_idx,ω_idx] = att_jac'*ForwardDiff.jacobian(f, v⁺[ω_idx])
        end
        
        # Check break condition
        res = norm([M*(v⁺-v) - J'*λ - F; c])
        # println("res: ", res)
        if norm(res) < 1e-12
            println("breaking at iter: $i")
            break
        end
        i == max_iters && @warn "Max iters reached"
        
        # Newton solve
        A = [M -J';
            -J*dq_dv zeros(nc, nc)]
        b = [M*v + F; 
            (c - J*dq_dv*v⁺)]
        sol = A\b  

        # Line search         
        v_old = copy(v⁺)
        dv = sol[1:nv]-v_old
        alpha = 1.0
        res_new = res + 1
        while (res_new > res) & (alpha > .5^10)
            v⁺ = v_old + alpha*dv
            q_next!(q⁺,v⁺,q,dt)
            c!(c,q⁺)
            J!(J,c!,q⁺,eltype(x))
            # F .= wrenches(model, [q⁺;v⁺], u) * dt
            F .= wrenches(model, x, u) * dt
            λ .= J'\(M*(v⁺-v) - F)
           
            res_new = norm([M*(v⁺-v) - J'*λ - F; c])
            # println("res new: ", res_new)
            alpha /= 2.0             
        end
        @assert (res_new < res)
    end

    return [q⁺; v⁺]
end

function discrete_jacobian(model::SimpleVine3D, x⁺, z, dt)
    n, m = size(model)  
    nb, nq, nv, nc = model.nb, model.nq, model.nv, model.nc
    M, J, c, c! = model.M, model.J, model.c, model.c!
    
    function f_imp(z)
        # Unpack
        q⁺ = z[1:nq]
        v⁺ = z[nq .+ (1:nv)]
        q = z[n .+ (1:nq)]
        v = z[n+nq .+ (1:nv)]
        u = z[2*n .+ (1:m)]
        λ = z[2*n+m .+ (1:nc)]

        F = wrenches(model, [q; v], u) * dt
        J = zeros(eltype(z),size(model.J))
        J!(J,c!,q⁺,eltype(z))
        q_next = zeros(eltype(z),size(q⁺))
        q_next!(q_next,v⁺,q,dt)

        return [M*(v⁺-v) - (J'*λ + F); q⁺ - q_next]
    end
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;z])
    ABC = -all_partials[:,1:n]\all_partials[:,1+n:end]
    # A = ABC[:, 1:n]
    # B = ABC[:, n .+ (1:m)]
    # C = ABC[:, n+m .+ (1:nc)]

    # multiply by att_jac
    att_jac = [Rotations.∇differential(UnitQuaternion(z[7*(i-1) .+ (1:4)]...)) for i=1:nb]
    att_jac⁺ = [Rotations.∇differential(UnitQuaternion(x⁺[7*(i-1) .+ (1:4)]...)) for i=1:nb]

    ABC′ = zeros(2*nv,n+m+nc)
    for i=1:nb
        # dr⁺/dz
        ABC′[6*(i-1) .+ (1:3), :] = ABC[7*(i-1) .+ (1:3), :]
        # dR⁺/dz
        ABC′[6*(i-1) .+ (4:6), :] = att_jac⁺[i]'*ABC[7*(i-1) .+ (4:7), :]
    end
    # dv⁺/dz
    ABC′[nv .+ (1:nv), :] = ABC[nq .+ (1:nv), :]

    A_big = ABC′[:, 1:n]
    B = ABC′[:, n .+ (1:m)]
    C = ABC′[:, n+m .+ (1:nc)]

    A = zeros(2*nv,2*nv)
    for i=1:nb
        # dx⁺/dr
        A[:, 6*(i-1) .+ (1:3)] =  A_big[:, 7*(i-1) .+ (1:3)]
        # dx⁺/dR
        A[:, 6*(i-1) .+ (4:6)] = A_big[:, 7*(i-1) .+ (4:7)]*att_jac[i]
    end
    # dx⁺/dv
    A[:, nv .+ (1:nv)] = A_big[:, nq .+ (1:nv)]
    
    J!(model.J,c!,x⁺[1:nq],Float64)
    G = [model.J zeros(nc, nv)]

    return A,B,C,G
end

# compute maximal coordinate configuration given body rotations
function generate_config(model, rotations)
    @assert model.nb == length(rotations)
    pin = zeros(3)
    q = zeros(0)   
    d = [0,0,-model.d]
    for i = 1:model.nb
        r = UnitQuaternion(rotations[i]...)
        delta = r * d
        q = [q; pin+delta; Rotations.params(r)]
        pin += 2*delta
    end
    return q
end