module MeshLoader 

export load_mesh

using FileIO
using GeometryBasics: GeometryBasics    # The one just for compatible
using Meshes                            # The one real used
using MeshBridge

function load_mesh(fname)
	old_mesh = load(fname; pointtype=GeometryBasics.Point{3, Float64})
	mesh = convert(Mesh, old_mesh)
end

end