using Rotations, LinearAlgebra, ForwardDiff

# q = UnitQuaternion(1,.1,.2,.3)
# Rotations.params(q)
# Rotations.∇differential(q)

# q0 = UnitQuaternion(RotX(.1))
# q1 = UnitQuaternion(RotX(.2))
# Rotations.rotation_error(q0, q1, Rotations.CayleyMap())
# ω⁺ = [.1,0,0]
# r⁺ = Rotations.params(Rotations.expm(ω⁺) * q0)
# rotation_angle(UnitQuaternion(r⁺))

#----------------------------------------------------
# nb = model.nb
# nv = model.nv
# nc = model.nc

# function q_aug(v⁺)
#     rotations = [RotX(.1) for i=1:model.nb]
#     q = generate_config(model, rotations)
#     q⁺ = ones(eltype(v⁺), size(q))
#     for i=1:nb
#         # position
#         r_idx = 7*(i-1) .+ (1:3)
#         q⁺[r_idx] = q[r_idx] + v⁺[6*(i-1) .+ (1:3)]*dt

#         # orientation
#         R_idx = 7*(i-1) .+ (4:7)
#         R = UnitQuaternion(q[R_idx])
#         ω⁺ = v⁺[6*(i-1) .+ (4:6)]
#         R⁺ = Rotations.params(Rotations.expm(ω⁺*dt) * R)
#         q⁺[R_idx] = R⁺/norm(R⁺)
#     end
#     return q⁺ 
# end

# function c_aug(v⁺)
#     q⁺ = q_aug(v⁺)
#     c = ones(eltype(v⁺), nc)
#     model.c!(c,q⁺)
#     return c
# end

# v⁺ = zeros(12)
# v⁺[7:9] .= 2
# q⁺ = q_aug(v⁺)
# dc_dq = ForwardDiff.jacobian(model.c!, ones(model.nc), q⁺)
# dq_dv = ForwardDiff.jacobian(q_aug, v⁺)
# dc_dv = ForwardDiff.jacobian(c_aug, v⁺)

# function manual_dc_dv(q,q⁺,v⁺)
#     c! = model.c!
#     J_big = ForwardDiff.jacobian(c!, ones(eltype(q⁺), nc), q⁺)
#     J = copy(model.J)
#     for i=1:nb
#         # dc/dr
#         J[:, 6*(i-1) .+ (1:3)] = J_big[:, 7*(i-1) .+ (1:3)]

#         # dc/dq
#         att_jac = Rotations.∇differential(UnitQuaternion(q⁺[7*(i-1) .+ (4:7)]))
#         J[:, 6*(i-1) .+ (4:6)] = J_big[:, 7*(i-1) .+ (4:7)] * att_jac
#     end

#     # dq_dv
#     dq_dv = Matrix(dt*I, nv, nv)
#     for i=1:nb            
#         R = UnitQuaternion(q[7*(i-1) .+ (4:7)])
#         R⁺ = UnitQuaternion(q⁺[7*(i-1) .+ (4:7)])
#         att_jac = Rotations.∇differential(R⁺)

#         ω_idx = 6*(i-1) .+ (4:6)
#         f(ω⁺) = Rotations.params(Rotations.expm(ω⁺*dt) * R)
#         dq_dv[ω_idx,ω_idx] = att_jac'*ForwardDiff.jacobian(f, v⁺[ω_idx])
#     end
#     return J, dq_dv
# end

# J_man, dq_dv_man = manual_dc_dv(q0,q0,v⁺)

# -----------------------------------------------
# function q_next!(q⁺,v⁺,q,dt)
#     nb = Int(length(q⁺)/7)
#     for i=1:nb
#         # position
#         r_idx = 7*(i-1) .+ (1:3)
#         q⁺[r_idx] = q[r_idx] + v⁺[6*(i-1) .+ (1:3)]*dt

#         # orientation
#         R_idx = 7*(i-1) .+ (4:7)
#         R = UnitQuaternion(q[R_idx])
#         ω⁺ = v⁺[6*(i-1) .+ (4:6)]
#         R⁺ = Rotations.params(Rotations.expm(ω⁺*dt) * R)
#         q⁺[R_idx] = R⁺/norm(R⁺)
#     end
# end

# function f_imp(z)
#     # Unpack
#     q⁺ = z[1:nq]
#     v⁺ = z[nq .+ (1:nv)]
#     q = z[n .+ (1:nq)]
#     v = z[n+nq .+ (1:nv)]
#     u = z[2*n .+ (1:m)]
#     λ = z[2*n+m .+ (1:nc)]

#     F = wrenches(model, [q; v], u) * dt
#     J = zeros(eltype(z),size(model.J))
#     J!(J,c!,q⁺,eltype(z))
#     q_next = zeros(eltype(z),size(q⁺))
#     q_next!(q_next,v⁺,q,dt)

#     return [M*(v⁺-v) - (J'*λ + F); q⁺ - q_next]
# end

# all_partials = ForwardDiff.jacobian(f_imp, [x⁺;z])
# ABC = -all_partials[:,1:n]\all_partials[:,1+n:end]

function f(z)    
    R = UnitQuaternion(z[1:4])
    ω⁺ = z[5:7]
    R⁺ = Rotations.params(Rotations.expm(ω⁺) * R)
    return [R⁺/norm(R⁺); ω⁺]
end

z0_1 = [1,0,0,0,.1,0,0]
R0_1 = UnitQuaternion(z0_1[1:4])

z0_2 = [.99,sqrt(1-.99^2),0,0,.2,0,0]
R0_2 = UnitQuaternion(z0_2[1:4])

dR0 = R0_2 ⊖ R0_1
dz0 = [dR0[:];(z0_2-z0_1)[5:7]]

zf_1 = f(z0_1)
Rf_1 = UnitQuaternion(zf_1[1:4])

zf_2 = f(z0_2)
Rf_2 = UnitQuaternion(zf_2[1:4])

att_jac = Rotations.∇differential(R0_1)
att_jac⁺ = Rotations.∇differential(Rf_1)

AB = ForwardDiff.jacobian(f, z0_1)
AB = [att_jac⁺'*AB[1:4,:]; AB[5:7,:]]
A = AB[:,1:4]*att_jac
B = AB[:,5:7]

dRf = (Rf_2 ⊖ Rf_1)[:]
@show dzf = [dRf;(zf_2-zf_1)[5:7]]
dzf-[A B]*dz0
