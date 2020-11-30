# Maximal dynamics
c!(x) = [x[1] - .5*cos(x[3]-pi/2); 
         x[2] - .5*sin(x[3]-pi/2)]

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

function getABCG2(x⁺,x,u,λ,dt)
    MInv = Matrix(I,3,3)
    q⁺ = x⁺[1:3]
    J = ForwardDiff.jacobian(c!, q⁺)
    
    A = 1.0*Matrix(I,6,6)
    B = [zeros(3,1);MInv*[0;0;dt]]
    C = [zeros(3,2); MInv*J']
    g = [0;0;0;MInv*[0;-9.81*dt;0]]

    Ahat = copy(A)
    Ahat[1:3,4:6] = -dt*Matrix(I,3,3)
    Ahat[6,6] += .1*dt

    A = Ahat\A
    B = Ahat\B
    C = Ahat\C
    g = Ahat\g

    G = [zeros(2,3) J]

    return A,B,C,G,g
end

th0 = rand()
x0 = [.5*cos(th0-pi/2); .5*sin(th0-pi/2); th0; zeros(3)]
u0 = [.1]
x1, L0 = f(x0,u0,dt)
x2, L1 = f(x1,2*u0,dt)
f_imp([x1;x0;u0;L0])
A,B,C,G = getABCG(x1,x0,u0,L0,dt)
(x2-x1) - (A*(x1-x0) + B*u0 + C*(L1-L0))

A,B,C,G,g = getABCG2(x1,x0,u0,L0,dt)
x1 - (A*x0 + B*u0 + C*L0 + g)