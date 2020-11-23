function getABCG(x⁺,x,u,λ,dt)
    function f_imp(z)
        # Unpack
        q⁺ = z[1:3]
        v⁺ = z[4:6]
        q = z[7:9]
        v = z[10:12]
        u = z[13]
        λ = z[14:end]
    
        b = 0.1
        J = ForwardDiff.jacobian(c!, q⁺)
        F = [0; -9.81; u-b*v⁺[3]] * dt
        return [(v⁺-v) - (J'*λ + F); q⁺ - (q + v⁺*dt)]
    end

    n = length(x)
    m = length(u)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:6]\all_partials[:,7:end]
    A = ABC[:, 1:n]
    B = ABC[:, n .+ (1:m)]
    C = ABC[:, n+m+1:end]

    c_imp(z) = c!(z[1:3])
    G = ForwardDiff.jacobian(c!, x⁺)

    return A,B,C,G
end

A,B,C,G = getABCG(X_max[:,2],X_max[:,1],U[:,1],zeros(2),dt)
