abstract type MaxImp <: Implicit end

############################################################################################
#                          IMPLICIT DISCRETE TIME METHODS                                  #
############################################################################################

@inline discrete_dynamics(::Type{Q}, model::AbstractModel, z::AbstractKnotPoint) where Q<:Implicit =
    discrete_dynamics(Q, model, state(z), control(z), z.t, z.dt)

@inline discrete_dynamics(::Type{Q}, model::AbstractModel, x, u, t, dt) where Q<:Implicit =
    integrate(Q, model, x, u, t, dt)

function propagate_dynamics(::Type{Q}, model::AbstractModel, z_::AbstractKnotPoint, z::AbstractKnotPoint) where Q<:Implicit
    x_next = discrete_dynamics(Q, model, z)
    set_state!(z_, x_next)
end

function discrete_jacobian!(::Type{Q}, ∇f, model::AbstractModel,
    z::AbstractKnotPoint{T,N,M}) where {T,N,M,Q<:Implicit}
    throw(ErrorException("Implicit discrete jacobian not implemented"))
    # ix,iu,idt = z._x, z._u, N+M+1
    # t = z.t
    # fd_aug(s) = discrete_dynamics(Q, model, s[ix], s[iu], t, z.dt)
    # ∇f .= ForwardDiff.jacobian(fd_aug, SVector{N+M}(z.z))
    # return nothing
end
