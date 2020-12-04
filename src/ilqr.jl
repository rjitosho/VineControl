using ForwardDiff
using LinearAlgebra
using Plots
using Random

# Maximal dynamics
function c!(x) 
    throw("Not implemented")
    return c
end

function f(x,u,dt)     
    throw("Not implemented")
    return [q⁺; v⁺], λ 
end

function getABCG(x⁺,x,u,λ,dt)
    function f_imp(z)
        throw("Not implemented")
        return [M*(v⁺-v) - (J'*λ + F); q⁺ - (q + v⁺*dt)]
    end

    throw("Not implemented")
    return A,B,C,G
end

function state_error(x,x0)
    throw("Not implemented")
    return dx
end

#iLQR
function rollout(x0,U,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    Lam = zeros(4,N-1)
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
        dx = state_error(X[:,k], xf)
        J += 0.5*dx'*Q*dx + 0.5*U[:,k]'*R*U[:,k]
    end
    dx = state_error(X[:,N], xf)
    J += 0.5*dx'*Qf*dx
    return J
end

function backwardpass(X,Lam,U,F,Q,R,Qf,xf)
    _, N = size(X)
    n,_ = size(Q)
    m = size(U,1)
    nc = 4
    
    S = zeros(n,n,N)
    s = zeros(n,N)    
    K = zeros(m,n,N-1)
    l = zeros(m,N-1)
    
    S[:,:,N] = Qf
    s[:,N] = Qf*state_error(X[:,N], xf)
    
    mu = 0.0
    k = N-1
    
    while k >= 1
        q = Q*state_error(X[:,k], xf)
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
    Lam = zeros(4,N-1)
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
    U = zeros(1,N-1)
    Lam = zeros(4,N-1)
    X[:,1] = x0
    for k = 1:N-1
        dx = state_error(X[:,k], xf)
        U[k] = u0-Ku*dx
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam, U
end
