using ForwardDiff
using LinearAlgebra
using Plots
using Random

## Simple Pendulum
n = 6 # number of states 
m = 1 # number of controls

#initial and goal conditions
x0 = zeros(6)
xf = zeros(6)

#costs
Q = zeros(n,n)
Q[3,3] = 0.3
Q[6,6] = 0.3
# Q = 0.3*Matrix(I,n,n)
Qf = 100*Q
R = 0.3*Matrix(I,m,m)

#simulation
dt = 0.03
tf = 5.0

# Maximal dynamics
c!(x) = [x[1] + .5*sin(x[3]); 
         (x[2]-.5) + .5*cos(x[3])]

function f(x,u,dt)
    m = 1.
    b = 0.1
    g = 9.81
    M = 1.0*Matrix(I,3,3)

    q = x[1:3]
    v = x[4:6]
    λ = zeros(2)

    q⁺ = copy(q)
    v⁺ = copy(v)

    max_iters = 10
    for i=1:max_iters      
        c = c!(q⁺)
        J = ForwardDiff.jacobian(c!, q⁺)
        F = [0; -m*g; u[1]-b*v⁺[3]] * dt

        # Check break condition
        f = M*(v⁺-v) - J'*λ - F
        # println(norm([f;c]))
        if norm([f;c]) < 1e-12
            # println("breaking at iter: $i")
            break
        end
        i == max_iters && @warn "Max iters reached"

        # Newton solve
        A = [M -J';
            -J zeros(2,2)]
        d = [M*v + F; 
            (c + J*(q-q⁺))/dt]
        sol = A\d
        
        # Update        
        v⁺ = sol[1:3]
        λ = sol[4:end]
        q⁺ = q + v⁺*dt
    end
    return [q⁺; v⁺], λ 
end

function getABCG(x⁺,x,u,λ,dt)
    function f_imp(z)
        # Unpack
        q⁺ = z[1:3]
        v⁺ = z[4:6]
        q = z[7:9]
        v = z[10:12]
        u = z[13]
        λ = z[14:end]
    
        M = 1.0*Matrix(I,3,3)
        b = 0.1
        J = ForwardDiff.jacobian(c!, q⁺)
        F = [0; -9.81; u-b*v⁺[3]] * dt
        return [M*(v⁺-v) - (J'*λ + F); q⁺ - (q + v⁺*dt)]
    end

    n = length(x)
    m = length(u)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:6]\all_partials[:,7:end]
    A = ABC[:, 1:n]
    B = ABC[:, n .+ (1:m)]
    C = ABC[:, n+m+1:end]

    # c_imp(z) = c!(z[1:3])
    # G = ForwardDiff.jacobian(c_imp, x⁺)

    J = ForwardDiff.jacobian(c!, x⁺[1:3])
    G = [zeros(2,3) J]
    return A,B,C,G
end

#iLQR
function rollout(x0,U,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    Lam = zeros(2,N-1)
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
      J += 0.5*(X[:,k] - xf)'*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N] - xf)'*Qf*(X[:,N] - xf)
    return J
end

function backwardpass(X,Lam,U,F,Q,R,Qf,xf)
    n, N = size(X)
    m = size(U,1)
    nc = 2
    
    S = zeros(n,n,N)
    s = zeros(n,N)    
    K = zeros(m,n,N-1)
    l = zeros(m,N-1)
    
    S[:,:,N] = Qf
    s[:,N] = Qf*(X[:,N] - xf)
    v1 = 0.0
    v2 = 0.0

    mu = 0.0
    k = N-1
    
    while k >= 1
        q = Q*(X[:,k] - xf)
        r = R*(U[:,k])
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
        @show Ku

        l_all = M\[r;zeros(nc)]
        lu = l_all[1:m,:]
        lλ = l_all[m+1:m+nc,:]
        l[:,k] = lu

        Abar = A-B*Ku-C*Kλ
        bbar = -B*lu - C*lλ
        S[:,:,k] = Q + Ku'*R*Ku + Abar'*S⁺*Abar
        s[:,k] = q - Ku'*r + Ku'*R*lu + Abar'*S⁺*bbar + Abar'*s⁺

        # terms for line search
        # v1 += (l[:,k]'*Qu[:,:,k])[1]
        # v2 += (l[:,k]'*Quu[:,:,k]*l[:,k])[1]
        
        k = k - 1;
    end
    return K, l, v1, v2
end

# function ricatti(F,Q,R,dt)
#     xg = zeros(6)
#     ug = [0.0]
#     _, Lamg = f(xg,ug,dt)
#     A,B,C,G,g = F(xg,xg,ug,Lamg,dt)

#     S = [Q]
#     s = [] 
#     K = []
#     l = []
    
#     while k >= 1
#         S⁺ = S[:,:,k+1]

#         D = B - C/(G*C)*G*B
#         M11 = R + D'*S⁺*B
#         M12 = D'*S⁺*C
#         M21 = G*B
#         M22 = G*C

#         M = [M11 M12;M21 M22]
#         b = [D'*S⁺;G]*A

#         K_all = M\b
#         Ku = K_all[1:m,:]
#         Kλ = K_all[m+1:m+nc,:]
#         K[:,:,k] = Ku

#         l_all = M\[D'*S⁺;G]*g
#         lu = l_all[1:m,:]
#         lλ = l_all[m+1:m+nc,:]
#         l[:,k] = lu

timesteps = 100
X = repeat(xf,outer=(1,timesteps+1))
Lam = repeat([0;0.2943],outer=(1,timesteps))
U = zeros(1,timesteps)
K, l, v1, v2 = backwardpass(X,Lam,U,getABCG,Q,R,Q,xf)
K6 = [K[1,6,i] for i=1:timesteps]
K3 = [K[1,3,i] for i=1:timesteps]
plot([K3 K6])
