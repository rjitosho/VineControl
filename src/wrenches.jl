function wrenches(model::SimpleVine3D, x, u)
    nb, nq, nv = model.nb, model.nq, model.nv

    q = x[1:nq]
    v = x[1+nq:end]

    # Gravity
    F = zeros(nv)
    F[3:6:end] .= -model.M[1,1]*9810

    # Rotation
    J = model.M[4:6,4:6]
    for i=1:nb
        ω_idx = 6*(i-1) .+ (4:6)
        ω = v[ω_idx]
        F[ω_idx] += - ω × (J*ω)
    end
    
    # Base pin
    err = UnitQuaternion(q[4:7]) ⊖ one(UnitQuaternion)
    F[4:6] += -model.K[1,1]*err.err -model.C[1,1]*v[4:6]

    # Other pins
    for i=1:nb-1
        ω1_idx = 6*(i-1) .+ (4:6)
        ω2_idx = 6*i .+ (4:6)

        # Spring
        R1 = UnitQuaternion(q[7*(i-1) .+ (4:7)])
        R2 = UnitQuaternion(q[7*i .+ (4:7)])
        err = R2 ⊖ R1
        F[ω1_idx] += model.K[1,1]*err.err
        F[ω2_idx] += -model.K[1,1]*err.err
        
        # Damping
        ω1 = v[ω1_idx]
        ω2 = v[ω1_idx]
        F[ω1_idx] += model.C[1,1]*(ω2-ω1)
        F[ω2_idx] += -model.C[1,1]*(ω2-ω1)
    end
    
    return F
end
