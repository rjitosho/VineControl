using Rotations

# q = UnitQuaternion(1,.1,.2,.3)
# Rotations.params(q)
# Rotations.∇differential(q)

# q0 = UnitQuaternion(RotX(.1))
# q1 = UnitQuaternion(RotX(.2))
# Rotations.rotation_error(q0, q1, Rotations.CayleyMap())
# ω⁺ = [.1,0,0]
# r⁺ = Rotations.params(Rotations.expm(ω⁺) * q0)
# rotation_angle(UnitQuaternion(r⁺))
nb = model.nb
nv = model.nv
nc = model.nc

function q_aug(v⁺)
    rotations = [RotX(.1) for i=1:model.nb]
    q = generate_config(model, rotations)
    q⁺ = ones(eltype(v⁺), size(q))
    for i=1:nb
        # position
        r_idx = 7*(i-1) .+ (1:3)
        q⁺[r_idx] = q[r_idx] + v⁺[6*(i-1) .+ (1:3)]*dt

        # orientation
        R_idx = 7*(i-1) .+ (4:7)
        R = UnitQuaternion(q[R_idx])
        ω⁺ = v⁺[6*(i-1) .+ (4:6)]
        R⁺ = Rotations.params(Rotations.expm(ω⁺*dt) * R)
        q⁺[R_idx] = R⁺/norm(R⁺)
    end
    return q⁺ 
end

function c_aug(v⁺)
    q⁺ = q_aug(v⁺)
    c = ones(eltype(v⁺), nc)
    model.c!(c,q⁺)
    return c
end
v⁺ = zeros(12)
v⁺[7:9] .= 2
q⁺ = q_aug(v⁺)
dc_dq = ForwardDiff.jacobian(model.c!, ones(model.nc), q⁺)
dq_dv = ForwardDiff.jacobian(q_aug, v⁺)
dc_dv = ForwardDiff.jacobian(c_aug, v⁺)

function manual_dc_dv(q,q⁺,v⁺)
    c! = model.c!
    J_big = ForwardDiff.jacobian(c!, ones(eltype(q⁺), nc), q⁺)
    J = copy(model.J)
    for i=1:nb
        # dc/dr
        J[:, 6*(i-1) .+ (1:3)] = J_big[:, 7*(i-1) .+ (1:3)]

        # dc/dq
        att_jac = Rotations.∇differential(UnitQuaternion(q⁺[7*(i-1) .+ (4:7)]))
        J[:, 6*(i-1) .+ (4:6)] = J_big[:, 7*(i-1) .+ (4:7)] * att_jac
    end

    # dq_dv
    dq_dv = Matrix(dt*I, nv, nv)
    for i=1:nb            
        R = UnitQuaternion(q[7*(i-1) .+ (4:7)])
        R⁺ = UnitQuaternion(q⁺[7*(i-1) .+ (4:7)])
        att_jac = Rotations.∇differential(R⁺)

        ω_idx = 6*(i-1) .+ (4:6)
        f(ω⁺) = Rotations.params(Rotations.expm(ω⁺*dt) * R)
        dq_dv[ω_idx,ω_idx] = att_jac'*ForwardDiff.jacobian(f, v⁺[ω_idx])
    end
    return J, dq_dv
end
J_man, dq_dv_man = manual_dc_dv(q0,q0,v⁺)