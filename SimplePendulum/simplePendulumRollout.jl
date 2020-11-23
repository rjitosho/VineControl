using ForwardDiff
using LinearAlgebra
using Plots

## Simple Pendulum
n = 2 # number of states 
m = 1 # number of controls

#initial and goal conditions
x0 = [0.; 0.]
xf = [pi; 0.] # (ie, swing up)

#costs
Q = 0.3*Matrix(I,n,n)
Qf = 30.0*Matrix(I,n,n)
R = 0.3*Matrix(I,m,m)

#simulation
dt = 0.03
tf = 5.0

# Maximal dynamics
c!(x) = [x[1] - .5*cos(x[3]-pi/2); x[2] - .5*sin(x[3]-pi/2)]

function fmax(x,u,dt)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    # I = 0.25
    g = 9.81
    M = 1.0*Matrix(I,3,3)

    q = x[1:3]
    v = x[4:6]
    λ = zeros(2)

    q⁺ = copy(q)
    v⁺ = copy(v)
    F = [0; -m*g; u[1]-b*v⁺[3]] * dt

    max_iters = 10
    for i=1:max_iters      
        c = c!(q⁺)
        J = ForwardDiff.jacobian(c!, q⁺)
        
        # Check break condition
        f = M*(v⁺-v) - J'*λ - F
        println(norm([f;c]))
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
      X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam
end

X_max, Lam = rollout([0;-.5;zeros(4)],1.08*U,fmax,dt,tf)
X_max, Lam = rollout([0;-.5;zeros(4)],U0,fmax,dt,tf)
P = plot(range(0,stop=tf,length=size(X_max,2)),X_max[3,:])
P = plot!(range(0,stop=tf,length=size(X_max,2)),X_max[6,:])
