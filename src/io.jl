module MeshLoader 

export load_mesh,
	   load_mesh_w_color

using FileIO
using FileIO: FileIO, @format_str, Stream, File, stream, skipmagic
using GeometryBasics: GeometryBasics    # The one just for compatible
using Meshes                            # The one real used
using MeshBridge
using ColorTypes

function load_mesh(fname)
	old_mesh = load(fname; pointtype=GeometryBasics.Point{3, Float64})
	mesh = convert(Mesh, old_mesh)
end

function load_mesh_w_color(fname) 
	old_mesh, colors = load_w_color(fname; pointtype=GeometryBasics.Point{3, Float64})
	mesh = convert(Mesh, old_mesh)
	return mesh, colors
end

function load_w_color(fname; element_types...)
	load_w_color(FileIO.query(fname); element_types...)
end

function load_w_color(fn::File{format}; element_types...) where{format}
    open(fn) do s
        skipmagic(s)
        load_w_color(s; element_types...)
    end
end

function load_w_color(fs::Stream{format"STL_ASCII"}; facetype=GeometryBasics.GLTriangleFace,
	pointtype=GeometryBasics.Point3f, normaltype=GeometryBasics.Vec3f, topology=false)
	#ASCII STL
	#https://en.wikipedia.org/wiki/STL_%28file_format%29#ASCII_STL
	io = stream(fs)

	points = pointtype[]
	faces = facetype[]
	normals = normaltype[]
	colors = RGB{Float32}[]

	vert_count = 0
	vert_idx = [0, 0, 0]

	face_color = RGB(0.0,0.0,0.0)

	while !eof(io)
		line = split(lowercase(readline(io)))
		if !isempty(line) && line[1] == "color"
			r, g, b = parse.(Float32, line[2:4])
			face_color = RGB(r, g, b)
		end

		if !isempty(line) && line[1] == "facet"
  			normal = normaltype(parse.(eltype(normaltype), line[3:5]))
  			readline(io) # Throw away outerloop
			for i in 1:3
				vertex = pointtype(parse.(eltype(pointtype),
									split(readline(io))[2:4]))
				if topology
					idx = findfirst(vertices(mesh), vertex)
				end
				if topology && idx != 0
					vert_idx[i] = idx
				else
					push!(points, vertex)
					push!(normals, normal)
					vert_count += 1
					vert_idx[i] = vert_count
				end
			end
			readline(io) # throwout endloop
			readline(io) # throwout endfacet
  			push!(faces, GeometryBasics.TriangleFace{Int}(vert_idx...))
			push!(colors, face_color)
		end
	end
	return GeometryBasics.Mesh(GeometryBasics.meta(points; normals=normals), faces), colors
end

end