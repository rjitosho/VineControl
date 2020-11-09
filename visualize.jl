using MeshCat, GeometryTypes, CoordinateTransformations, Colors

function change_units_Z(vine, Z; scale = .001)
    _, N = size(Z)
    return Z .* repeat([scale;scale;1],outer = (2*vine.nb,N))
end

function visualize!(m::SimpleVine,Z)
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
        if mod(i,2) == 1
            setobject!(vis["pp$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
            setobject!(vis["pd$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
        else
            setobject!(vis["pp$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
            setobject!(vis["pd$i"], HyperSphere(GeometryTypes.Point(0.,0.,0.),diam/2), MeshPhongMaterial(color=R))
        end
    end

    anim = MeshCat.Animation(Int(1/m.Δt))
    for k = 1:N
        atframe(anim, (k-1)) do
            for i = 1:nb
                x = Z[3*i-2,k]
                y = Z[3*i-1,k]
                θ = Z[3*i,k]
                settransform!(vis["pp$i"], Translation(x - d*cos(θ), y - d*sin(θ), 0))
                settransform!(vis["pd$i"], Translation(x + d*cos(θ), y + d*sin(θ), 0))
            end
        end
    end

    setanimation!(vis, anim)
end
