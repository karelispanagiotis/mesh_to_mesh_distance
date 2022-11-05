include("../src/io.jl")
include("../src/skd_tree.jl")

using CoordinateTransformations
using Random
using CSV
using DataFrames
using Meshes
using StaticArrays
using LinearAlgebra
using Base.Threads 
using BenchmarkTools
using .MeshLoader
using .SKDTree

function apply_tranform(transform::Transformation, obj::Mesh)
    new_points = transform.(vertices(obj))
    return SimpleMesh(new_points, topology(obj))
end

function getrange(N) 
    work = ceil(Int, N/nthreads())
    lo = 1 + (threadid()-1)*work
    hi = min(threadid()*work, N)
    return lo:hi
end

function preprocess_one_tree(trias)
    return sKDTree(trias)
end

function search_one_tree(tree, trias2)
    g_mindist = zeros(coordtype(eltype(trias2)), nthreads())
    g_tid1 = zeros(Int, nthreads()) 
    g_tid2 = zeros(Int, nthreads())
    @threads for t in 1:nthreads()
        mindist, id1, id2 = typemax(coordtype(eltype(trias2))), typemax(Int), typemax(Int)
        for i in getrange(length(trias2))
            (mindist, id1), id2 = min( ((mindist,id1),id2), (nearest_neighbour(tree,trias2[i]; radius²=mindist),i) )
        end
        g_mindist[threadid()] = mindist
        g_tid1[threadid()] = id1
        g_tid2[threadid()] = id2 
    end
    i = argmin(g_mindist)
    return g_mindist[i], g_tid1[i], g_tid2[i]
end

function alg_tree_queries(trias1, trias2)
    if length(trias1) < length(trias2)
        trias1, trias2 = trias2, trias1 
    end
    b1 = @benchmark(preprocess_one_tree($trias1), seconds=2)
    push!(preprocess_time_one_tree, mean(b1).time * 1e-6)
    tree = preprocess_one_tree(trias1)

    b2 = @benchmark(search_one_tree($tree, $trias2), 
            setup=(shuffle!($trias2)), seconds=2)
    push!(search_time_one_tree, mean(b2).time * 1e-6)
    return search_one_tree(tree, trias2)
end

function preprocess_two_trees(trias1, trias2)
    task = Threads.@spawn sKDTree(trias1)
    tree2 = sKDTree(trias2; leafsize=1)
    tree1 = fetch(task)
    return tree1, tree2
end

function alg_two_trees(trias1, trias2)
    b1 = @benchmark(preprocess_two_trees($trias1, $trias2), seconds=2)
    push!(preprocess_time_two_trees, mean(b1).time * 1e-6)

    tree1, tree2 = preprocess_two_trees(trias1, trias2)
    b2 = @benchmark(nearest_neighbours($tree1, $tree2), seconds=2)
    push!(search_time_two_trees, mean(b2).time * 1e-6)
    return nearest_neighbours(tree1, tree2)
end

benchmark_name = "airplanes"
filepath1 = "testcases/airplanes/obj1_topairplane_highres.obj"
filepath2 = "testcases/airplanes/obj2_bottomairplane_highres.obj"

mesh1 = load_mesh(filepath1)
mesh2 = load_mesh(filepath2)

# dir = SVector(0,1,0) #for scooby_distance, bunny_distance
dir = SVector(0,0,5)    

mesh2_aabb = boundingbox(mesh2)
const normalization_constant = norm(maximum(mesh2_aabb) - minimum(mesh2_aabb))

dist = 0.0
preprocess_time_one_tree = Vector{Float64}(undef, 0)
search_time_one_tree = Vector{Float64}(undef, 0)

preprocess_time_two_trees = Vector{Float64}(undef, 0)
search_time_two_trees = Vector{Float64}(undef, 0)

normal_distances = Vector{Float64}(undef, 0)
while(dist/normalization_constant < 1000.0)
    trias1 = collect(elements(mesh1))
    trias2 = collect(elements(mesh2))

    shuffle!(trias1)
    shuffle!(trias2)

    global dist, = alg_tree_queries(trias1, trias2) 
    alg_two_trees(trias1, trias2)


    dist = sqrt(dist)
    println(dist/normalization_constant)
    push!(normal_distances, dist/normalization_constant)
    
    global dir *= 1.1
    global mesh2 = apply_tranform(Translation(dir), mesh2)
end

times_one_tree = preprocess_time_one_tree .+ search_time_one_tree
times_two_trees = preprocess_time_two_trees .+ search_time_two_trees

df = DataFrame(NormDist=normal_distances, 
                preproc_one_tree = preprocess_time_one_tree,
                search_one_tree = search_time_one_tree,
                preproc_two_trees = preprocess_time_two_trees,
                search_two_trees = search_time_two_trees,
                one_tree_time_ms=times_one_tree, two_trees_time_ms=times_two_trees)


output_filename = string(benchmark_name, "_", nthreads(),"threads_time_vs_dist.csv")
CSV.write(output_filename, df)