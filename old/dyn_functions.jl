function f(z)
    x = z[1:model.n]
    u = z[model.n+1:end]

    m = model
    nc = m.nc
    nq = m.nq

    M = Matrix(m.M)
    MI = m.MInv

    q = x[1:nq]
    v = x[nq+1:end]

    J = zeros(eltype(x), size(m.J))
    c = zeros(eltype(x), size(m.c))
    c! = m.c!

    # assemble vector of angles and angular velocities for calculating fExt
    θall = q[3:3:end] - [m.θ0; q[3:3:end-3]]
    θdall = v[3:3:end] - [0; v[3:3:end-3]]

    # external impulse
    F = -m.R * (m.k * [θall; θdall]) # spring damper
    F[3:3:end] += u # bending actuation
    F *= dt # compute impulse

    # set initial guess to q_k, v_k
    v⁺ = copy(v)
    q⁺ = copy(q)
    λ = zeros(eltype(x), size(m.λ))

    for i=1:3
        # joint constraints
        c!(c,q⁺)
        ForwardDiff.jacobian!(J, c!, ones(eltype(x), nc), q⁺)

        # Newton solve
        A = [M -J';
            -J zeros(nc, nc)]
        b = [M*v + F; 
            (c + J*(q-q⁺))/dt]

        # unpack
        sol = A\b
        v⁺ = sol[1:nq]
        λ = sol[nq+1:end]
        q⁺ = q + v⁺*dt
    end
    
    # println(maximum(abs.(c)))

    if eltype(x) <: Float64
        m.J .= J
        m.c .= c
        m.λ .= λ
        println(λ)
    end
    return M*(v⁺-v)
    return (M*(v⁺-v) - J'*λ - F)
end

J_fd = zeros(nq, n+m)
ForwardDiff.jacobian!(J_fd, f, [x0; u0])
# println("done")