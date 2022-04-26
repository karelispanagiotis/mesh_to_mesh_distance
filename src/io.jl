using Meshes : SimpleMesh
using PlyIO: load_ply

function load_meshes(fname)
  ply = load_ply(fname)
  x = ply["vertex"]["x"]
  y = ply["vertex"]["y"]
  z = ply["vertex"]["z"]
  points = Point3f.(x, y, z)

  
  indices1 = findall(ply["face"]["obj_id"].==1)
  indices2 = findall(ply["face"]["obj_id"].==2)
  connec1 = [connect(Tuple(c.+1)) for c in ply["face"]["vertex_indices"][indices1]]
  connec2 = [connect(Tuple(c.+1)) for c in ply["face"]["vertex_indices"][indices2]]
  SimpleMesh(points, connec1), SimpleMesh(points, connec2)
end