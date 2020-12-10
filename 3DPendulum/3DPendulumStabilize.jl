using ForwardDiff
using LinearAlgebra
using Plots
using Rotations
using Random

## Simple Pendulum
n = 12 # number of states 
m = 1 # number of controls

#initial and goal conditions
R0 = UnitQuaternion(.9999,.0001,0, 0)
x0 = [R0*[0.; 0.; -.5]; Rotations.params(R0); zeros(6)]
xf = [0.; 0.;  .5; 0; 1; 0; 0; zeros(6)]

#costs
Q = .1*Diagonal([ones(3); zeros(4); ones(6)])
Qf = 100*Q
R = 0.1*Matrix(I,m,m)
cost_w = .1
cost_wf = 10.

#simulation
dt = 0.01
tf = 5.0

# Maximal dynamics
c!(x) = [x[1:3] - UnitQuaternion(x[4:7]...) * [0;0;-.5];x[6:7]]

function J!(q⁺)
    J_big = ForwardDiff.jacobian(c!, q⁺)
    R⁺ = UnitQuaternion(q⁺[4:7]...)
    att_jac⁺ = Rotations.∇differential(R⁺)
    return [J_big[:,1:3] J_big[:,4:7]*att_jac⁺]
end

nq = 7
nv = 6
nc = 5

function f(x,u,dt)
    m = 1.
    b = 0.1
    g = 9.81
    M = 1.0*Matrix(I,nv,nv)

    q = x[1:nq]
    v = x[nq+1:end]
    λ = zeros(nc)

    q⁺ = copy(q)
    v⁺ = copy(v)

    max_iters = 100
    for i=1:max_iters      
        c = c!(q⁺)
        J = J!(q⁺)
        F = [0;0;-m*g; u[1]-b*v⁺[4];0;0] * dt

        # Check break condition
        f = M*(v⁺-v) - J'*λ - F
        # println(norm([f;c]))
        if norm([f;c]) < 1e-12
            # println("breaking at iter: $i")
            break
        end
        i == max_iters && @warn "Max iters reached"

        quat⁺(ω⁺) = Rotations.params(Rotations.expm(ω⁺*dt) * UnitQuaternion(q[4:7]...))
        dq_dv = Matrix(dt*I, nv, nv)
        att_jac⁺ = Rotations.∇differential(UnitQuaternion(q⁺[4:7]...))
        dq_dv[4:6,4:6] = att_jac⁺'*ForwardDiff.jacobian(quat⁺, v⁺[4:6])
        
        # Newton solve
        A = [M -J';
            -J*dq_dv zeros(nc, nc)]
        d = [M*v + F; 
            (c - J*dq_dv*v⁺)]
        sol = A\d
        
        # Update        
        v⁺ = sol[1:nv]
        λ = sol[nv+1:end]
        q⁺ = [q[1:3]+v⁺[1:3]*dt; quat⁺(v⁺[4:6])]
    end
    return [q⁺; v⁺], λ 
end

function getABCG(x⁺,x,u,λ,dt)
    function f_imp(z)
        # Unpack
        q⁺ = z[1:nq]
        v⁺ = z[nq .+ (1:nv)]
        q = z[nq+nv .+ (1:nq)]
        v = z[2*nq+nv .+ (1:nv)]
        u = z[2*(nq+nv) .+ (1:m)]
        λ = z[2*(nq+nv)+m .+ (1:nc)]
    
        M = 1.0*Matrix(I,6,6)
        b = 0.1
        
        J = J!(q⁺)
        F = [0;0;-9.81; u[1]-b*v⁺[4];0;0] * dt

        ω⁺ = v⁺[4:6]
        quat⁺ = Rotations.params(Rotations.expm(ω⁺*dt) * UnitQuaternion(q[4:7]...))

        return [M*(v⁺-v) - (J'*λ + F); q⁺ - [q[1:3]+v⁺[1:3]*dt; quat⁺]]
    end

    n = length(x)
    m = length(u)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:n]\all_partials[:,n+1:end]

    att_jac = Rotations.∇differential(UnitQuaternion(x[4:7]...))
    att_jac⁺ = Rotations.∇differential(UnitQuaternion(x⁺[4:7]...))

    ABC′ = zeros(2*nv,n+m+nc)
    ABC′[1:3, :] = ABC[1:3, :]
    ABC′[4:6, :] = att_jac⁺'*ABC[4:7, :]
    ABC′[nv .+ (1:nv), :] = ABC[nq .+ (1:nv), :]

    A_big = ABC′[:, 1:(nq+nv)]
    B = ABC′[:, nq+nv .+ (1:m)]
    C = ABC′[:, nq+nv+m .+ (1:nc)]

    A = zeros(2*nv,2*nv)
    A[:, 1:3] =  A_big[:, 1:3]
    A[:, 4:6] = A_big[:, 4:7]*att_jac
    A[:, nv .+ (1:nv)] = A_big[:, nq .+ (1:nv)]

    J = J!(x⁺)
    G = [J zeros(size(J))]
    return A,B,C,G
end


function state_error(x,x0)
    err = zeros(2*nv)
    dx = x-x0
    for i=1:1
        err[6*(i-1) .+ (1:3)] = dx[7*(i-1) .+ (1:3)]
        dq = UnitQuaternion(x[7*(i-1) .+ (4:7)]...) ⊖ UnitQuaternion(x0[7*(i-1) .+ (4:7)]...)
        err[6*(i-1) .+ (4:6)] = dq[:]
    end
    err[nv .+ (1:nv)] = dx[nq .+ (1:nv)]
    return err
end

#iLQR
function rollout(x0,U,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    Lam = zeros(nc,N-1)
    X[:,1] = x0
    for k = 1:N-1
        # print("k = $k ")
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam
end

function cost(X,U,Q,R,Qf,xf)
    N = size(X,2)
    J = 0.0
    for k = 1:N-1
        # dx = state_error(X[:,k], xf)
        dx = X[:,k] - xf
        J += 0.5*dx'*Q*dx + 0.5*U[:,k]'*R*U[:,k]
        q = X[4:7,k]
        dq = xf[4:7]'q
        J += .1*min(1+dq, 1-dq)
    end
    # dx = state_error(X[:,N], xf)
    dx = X[:,N] - xf
    J += 0.5*dx'*Qf*dx
    q = X[4:7,N]
    dq = xf[4:7]'q
    J += .1*min(1+dq, 1-dq)
    return J
end

function compute_Qq(Q, x, xf)
    n = 12
    Q_ = Q[1,1]*Matrix(I,n,n)
    Q_[4:6,4:6] = abs(xf[4:7]'x[4:7])*Matrix(I,3,3)
    
    q_ = Q*(x - xf)
    deleteat!(q_,4)
    att_jac = Rotations.∇differential(UnitQuaternion(x[4:7]))
    q_[4:6] = att_jac'*xf[4:7]
    return Q_, q_
end

function backwardpass(X,Lam,U,F,Q,R,Qf,xf)
    Q_og = Q
    _, N = size(X)
    n = 12
    m = size(U,1)

    S = zeros(n,n,N)
    s = zeros(n,N)    
    K = zeros(m,n,N-1)
    l = zeros(m,N-1)
    
    S[:,:,N], s[:,N] = compute_Qq(Qf, X[:,N], xf)
    
    mu = 0.0
    k = N-1
    
    while k >= 1
        Q, q = compute_Qq(Q_og, X[:,k], xf)
        r = R*U[:,k]
        S⁺ = S[:,:,k+1]
        s⁺ = s[:,k+1]
        
        A,B,C,G = F(X[:,k+1],X[:,k],U[:,k],Lam[:,k],dt)
        
        D = B - C/(G*C)*G*B
        M11 = R + D'*S⁺*B
        M12 = D'*S⁺*C
        M21 = G*B
        M22 = G*C

        M = [M11 M12;M21 M22]
        b = [D'*S⁺;G]*A

        K_all = M\b
        Ku = K_all[1:m,:]
        Kλ = K_all[m+1:m+nc,:]
        K[:,:,k] = Ku

        l_all = M\[r + D'*s⁺; zeros(nc)]
        lu = l_all[1:m,:]
        lλ = l_all[m+1:m+nc,:]
        l[:,k] = lu

        Abar = A-B*Ku-C*Kλ
        bbar = -B*lu - C*lλ
        S[:,:,k] = Q + Ku'*R*Ku + Abar'*S⁺*Abar
        s[:,k] = q - Ku'*r + Ku'*R*lu + Abar'*S⁺*bbar + Abar'*s⁺

        k = k - 1;
    end
    return K, l
end

function forwardpass(X,U,f,J,K,l)
    N = size(X,2)
    m = size(U,1)
    Lam = zeros(nc,N-1)
    X_prev = copy(X)
    J_prev = copy(J)
    U_ = zeros(m,N-1)
    J = Inf
    dJ = 0.0
    
    alpha = 1.0
    while J > J_prev
        for k = 1:N-1
            dx = state_error(X[:,k], X_prev[:,k])
            U_[:,k] = U[:,k] - K[:,:,k]*dx - alpha*l[:,k]
            X[:,k+1], Lam[:,k] = f(X[:,k],U_[:,k],dt);
        end

        J = cost(X,U_,Q,R,Qf,xf)
        dJ = J_prev - J
        alpha = alpha/2.0;
    end

    println("New cost: $J")
    println("- Line search iters: ", abs(log(.5,alpha)))
    println("- Actual improvement: $(dJ)")
    return X, U_, J, Lam
end


function solve(x0,m,f,F,Q,R,Qf,xf,dt,tf,iterations=100,eps=1e-5;control_init="random")
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    
    if control_init == "random"
        Random.seed!(0)
        U = 5.0*rand(m,N-1)
    else
        U = zeros(m,N-1)
    end
    U0 = copy(U)
        
    X, Lam = rollout(x0,U,f,dt,tf)
    X0 = copy(X)
    Lam0 = copy(Lam)
    J_prev = cost(X,U,Q,R,Qf,xf)
    println("Initial Cost: $J_prev\n")
    
    K = zeros(2,2,2)
    l = zeros(2,2)
    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, l = backwardpass(X,Lam,U,F,Q,R,Qf,xf)
        X, U, J, Lam = forwardpass(X,U,f,J_prev,K,l)

        if abs(J-J_prev) < eps
          println("-----SOLVED-----")
          println("eps criteria met at iteration: $i")
          break
        end
        J_prev = copy(J)
    end
    
    return X, U, K, l, X0, U0, Lam0
end

function stable_rollout(Ku,x0,u0,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    U = zeros(m,N-1)
    Lam = zeros(nc,N-1)
    X[:,1] = x0
    for k = 1:N-1
        dx = state_error(X[:,k], xf)
        U[:,k] = u0-Ku*dx
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam, U
end

# test dynamics and jacobian
_, λ0 = f(xf,[0.],dt)
# A,B,C,G = getABCG(xf,xf,[0.],λ0,dt)

# STABILZE
timesteps = 300
X = repeat(xf,outer=(1,timesteps+1))
Lam = repeat(λ0,outer=(1,timesteps))
U = zeros(1,timesteps)
K, l = backwardpass(X,Lam,U,getABCG,Q,R,Q,xf)

# julia> K[:,:,1]
# 1×12 Array{Float64,2}:
#  0.0  -6.96621  0.0  13.8464  0.0  0.0  0.0  -2.80976  0.0  2.80976  0.0  0.0

K1 = [K[1,2,i] for i=1:timesteps]
K2 = [K[1,4,i] for i=1:timesteps]
plot([K1 K2])

Ku = K[:,:,1]
x1, _ = f(xf,[1.],dt)
X, Lam, U=stable_rollout(Ku,x1,U[:,1],f,dt,tf)
plot(X[2,:])

# SWING UP
X, U, K, l, X0, U0, Lam0 = solve(x0,m,f,getABCG,Q,R,Qf,xf,dt,tf,1,control_init="random");
_,N = size(X)

# Kth = [K[1,4,i] for i=1:N-1]
# Kthd = [K[1,10,i] for i=1:N-1]
# plot([Kth Kthd])
# plot(Kthd)

quats = [UnitQuaternion(X[4:7,i]) for i=1:N]
angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
plot(angles)
plot!(X[10,:])
# plot(U[:])

# R1 = UnitQuaternion(1,0,0,0) 
# rotation_angle(R1)
# R2 = UnitQuaternion(.5,.5,0,0)
# rotation_angle(R2)
# R3 = UnitQuaternion(.5,1,0,0)
# rotation_angle(R3)

# e21 = R2 ⊖ R1
# e31 = R3 ⊖ R1
# e32 = R3 ⊖ R2

# e31
# inv(CayleyMap())(UnitQuaternion(e21) * UnitQuaternion(e32))
