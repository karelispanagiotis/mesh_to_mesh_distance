include("../src/io.jl")

using Meshes
using CSV
using DataFrames
using BenchmarkTools
using Random
using Base.Threads
using .MeshLoader

include("../src/primitive_distances.jl")
include("../src/skd_tree.jl")
using Meshes
using Base.Threads
using .PrimitiveDistances
using .SKDTree

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
    b1 = @benchmark(preprocess_one_tree($trias1))
    push!(preprocess_time_one_tree, mean(b1).time * 1e-6)
    tree = preprocess_one_tree(trias1)

    b2 = @benchmark(search_one_tree($tree, $trias2), 
            setup=(shuffle!($trias2)), samples=2, evals=10, seconds=10)
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
    b1 = @benchmark(preprocess_two_trees($trias1, $trias2))
    push!(preprocess_time_two_trees, mean(b1).time * 1e-6)

    tree1, tree2 = preprocess_two_trees(trias1, trias2)
    b2 = @benchmark(nearest_neighbours($tree1, $tree2))
    push!(search_time_two_trees, mean(b2).time * 1e-6)
    return nearest_neighbours(tree1, tree2)
end

dir = ARGS[1]

all_files = readdir(joinpath("testcases/", dir), join=true)
obj1_files = all_files[1:3]
obj2_files = all_files[4:6]

n = Vector{Int32}(undef, 0)
preprocess_time_one_tree = Vector{Float64}(undef, 0)
search_time_one_tree = Vector{Float64}(undef, 0)

preprocess_time_two_trees = Vector{Float64}(undef, 0)
search_time_two_trees = Vector{Float64}(undef, 0)

for i in 1:3
    for j in 1:3 
        mesh1 = load_mesh(obj1_files[i])
        mesh2 = load_mesh(obj2_files[j])

        trias1 = collect(elements(mesh1))
        trias2 = collect(elements(mesh2))

        shuffle!(trias1)
        shuffle!(trias2)

        push!(n, length(trias1) + length(trias2))
        
        alg_tree_queries(trias1, trias2) 
        alg_two_trees(trias1, trias2)
    end
end

times_one_tree = preprocess_time_one_tree .+ search_time_one_tree
times_two_trees = preprocess_time_two_trees .+ search_time_two_trees

df = DataFrame(N_total=n, preproc_time_one=preprocess_time_one_tree, 
                search_time_one=search_time_one_tree,
                preproc_time_two = preprocess_time_two_trees,
                search_time_two = search_time_two_trees, 
                one_tree_time_ms=times_one_tree, two_trees_time_ms=times_two_trees)

sort!(df)

filename = string(ARGS[1], "_", nthreads(), "threads_exec_time.csv")
CSV.write(filename, df)