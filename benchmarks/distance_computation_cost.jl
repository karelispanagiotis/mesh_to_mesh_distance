include("../src/io.jl")
include("../src/primitive_distances.jl")

using BenchmarkTools
using Meshes
using .MeshLoader
using .PrimitiveDistances

function pairwise_distance_boxes(boxes1, boxes2)
    dd = zero(coordtype(eltype(points1)))
    for b1 ∈ boxes1 
        for b2 ∈ boxes2 
            dd = AABB_distance²(b1, b2)
        end
    end
    return dd
end

function pairwise_distance_triangles(trias1, trias2)
    dd = zero(coordtype(eltype(points1)))
    for t1 ∈ trias1 
        for t2 ∈ trias2 
            dd, = triangle_distance²(t1, t2)
        end
    end
end

function pairwise_distance_points(points1, points2)
    dd = zero(coordtype(eltype(points1)))
    for p1 ∈ points1 
        for p2 ∈ points2 
            V  = p2 - p1 
            dd = V⋅V
        end
    end
    return dd
end

function pairwise_distance_segments(segments1, segments2)
    for s1 ∈ segments1 
        for s2 ∈ segments2 
            PQP_SegPoints(s1, s2)
        end
    end
end

obj1 = load_mesh("testcases/airplanes_STL/obj1_topplane_lowres.obj");
obj2 = load_mesh("testcases/airplanes_STL/obj2_bottomplane_lowres.obj");

trias1 = collect(elements(obj1));
trias2 = collect(elements(obj2)); 

boxes1 = boundingbox.(trias1);
boxes2 = boundingbox.(trias2);

points1 = rand(Point3, length(trias1));
points2 = rand(Point3, length(trias2));

segments1 = [Segment(rand(Point3), rand(Point3)) for i=1:length(trias1)]
segments2 = [Segment(rand(Point3), rand(Point3)) for i=1:length(trias2)]

println("Triangle-Triangle Test:")
@btime pairwise_distance_triangles($trias1, $trias2)
println("Total Tests = ", length(trias1)*length(trias2))

println("Segment-Segment Test:")
@btime pairwise_distance_segments($segments1, $segments1)
println("Total Tests = ", length(segments1)*length(segments2))

println("Box-Box Test:")
@btime pairwise_distance_boxes($boxes1, $boxes2)
println("Total Tests = ", length(boxes1)*length(boxes2))

println("Point-Point Test:")
@btime pairwise_distance_points($points1, $points2)
println("Total Tests = ", length(points2)*length(points1))