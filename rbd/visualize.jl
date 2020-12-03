using MeshCat, GeometryTypes, GeometryBasics, CoordinateTransformations, Colors
# using GeometryBasics: HyperRectangle, HyperSphere, Vec, Point, Mesh

function change_units_Z(vine, Z; scale = .001)
    _, N = size(Z)
    return Z .* repeat([scale;scale;1],outer = (2*vine.nb,N))
end

function change_units_Z(vine::SimpleVine3D, Z; scale = .001)
    _, N = size(Z)
    return Z[1:vine.nq,:] .* repeat([scale*ones(3);ones(4)],outer = (vine.nb,N))
end

# function visualize!(m::SimpleVine,Z,dt)
#     _, N = size(Z)
#     diam = m.diam/1000 # tube diameter
#     d = m.d/1000
#     nb = m.nb

#     vis = Visualizer()
#     open(vis)

#     # create spheres for proximal and distant end point of each body
#     R = RGBA(1, 0, 0, 1.0)
#     B = RGBA(0, 0, 1, 1.0)
#     for i = 1:nb
#         if mod(i,2) == 1
#             setobject!(vis["pp$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
#             setobject!(vis["pd$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
#         else
#             setobject!(vis["pp$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
#             setobject!(vis["pd$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
#         end
#     end

#     anim = MeshCat.Animation(Int(1/dt))
#     for k = 1:N
#         atframe(anim, (k-1)) do
#             for i = 1:nb
#                 x = Z[3*i-2,k]
#                 y = Z[3*i-1,k]
#                 θ = Z[3*i,k]
#                 settransform!(vis["pp$i"], Translation(x - d*cos(θ), y - d*sin(θ), 0))
#                 settransform!(vis["pd$i"], Translation(x + d*cos(θ), y + d*sin(θ), 0))
#             end
#         end
#     end

#     setanimation!(vis, anim)
# end

function visualize!(m::SimpleVine3D,Z,dt)
    _, N = size(Z)
    diam = m.diam/1000 # tube diameter
    d = m.d/1000
    nb = m.nb

    vis = Visualizer()
    open(vis)

    # create spheres for proximal and distant end point of each body
    R = RGBA(1, 0, 0, 1.0)
    B = RGBA(0, 0, 1, 1.0)
    for i = 1:nb
        setobject!(vis["pp$i"], GeometryTypes.HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
        setobject!(vis["pd$i"], GeometryTypes.HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
    end

    anim = MeshCat.Animation(Int(1/dt))
    for k = 1:N
        atframe(anim, (k-1)) do
            for i = 1:nb
                CoM = Z[7*(i-1) .+ (1:3), k]
                r = UnitQuaternion(Z[7*(i-1) .+ (4:7), k]...)
                p1 = CoM - r * [0,0,-d]
                p2 = CoM + r * [0,0,-d]
                settransform!(vis["pp$i"], Translation(p1...))
                settransform!(vis["pd$i"], Translation(p2...))
            end
        end
    end

    setanimation!(vis, anim)
end
