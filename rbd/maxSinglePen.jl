using ForwardDiff
using LinearAlgebra
using Plots

# Maximal dynamics
c!(x) = [x[1] - .5*sin(x[3]); x[2] + .5*cos(x[3])]

function fmax(x,u,dt)
    g = 9.81
    M = [1.0       0.0   0.0;
        0.0        1.0   0.0;
        0.0        0.0   0.0841667]

    q = x[1:3]
    v = x[4:6]
    λ = zeros(2)

    q⁺ = copy(q)
    v⁺ = copy(v)
    F = [0; -g; 0] * dt

    max_iters = 10
    for i=1:max_iters      
        c = c!(q⁺)
        J = ForwardDiff.jacobian(c!, q⁺)
        
        # Check break condition
        f = M*(v⁺-v) - J'*λ - F
        # println(norm([f;c]))
        if norm([f;c]) < 1e-12
            println("breaking at iter: $i")
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

function rollout(x0,U,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    Lam = zeros(2,N-1)
    X[:,1] = x0
    for k = 1:N-1
        print("iter $k\t")
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam
end

#simulation
dt = 0.001
tf = 4.0
N = ceil(Int,tf/dt)
U = zeros(1,N)

X_max, Lam = rollout([.5*sin(pi/2);-.5*cos(pi/2);pi/2;zeros(3)],U,fmax,dt,tf)
P = plot(range(0,stop=tf,length=size(X_max,2)),X_max[1,:])
P = plot(range(0,stop=tf,length=size(X_max,2)),X_max[3,:])
