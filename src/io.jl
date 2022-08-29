module MeshLoader 

export load_mesh

using FileIO
using Meshes
using MeshBridge

function load_mesh(fname)
	old_mesh = load(fname)
	mesh = convert(Mesh, old_mesh)
end

end