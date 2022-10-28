include("../src/distance_algorithms.jl")
include("../src/io.jl")

using Meshes
using CSV
using DataFrames
using BenchmarkTools
using Random
using Base.Threads
using .MeshLoader
using .DistanceAlgorithms

dir = ARGS[1]

dir = ARGS[1]
all_files = readdir(joinpath("testcases/", dir), join=true)
obj1_files = all_files[1:3]
obj2_files = all_files[4:6]

n1 = Vector{Int32}(undef, 0)
n2 = Vector{Int32}(undef, 0)
times_one_tree = Vector{Float64}(undef, 0)
times_two_trees = Vector{Float64}(undef, 0)

for i in 1:3
    for j in 1:3 
        mesh1 = load_mesh(obj1_files[i])
        mesh2 = load_mesh(obj2_files[j])

        trias1 = collect(elements(mesh1))
        trias2 = collect(elements(mesh2))

        push!(n1, length(trias1))
        push!(n2, length(trias2))
        
        b1 = @benchmark(alg_tree_queries($trias1, $trias2), 
                        setup=(shuffle!($trias2)), samples=10, evals=50, seconds=15)
        push!(times_one_tree, mean(b1).time * 1e-6)
        
        b2 = @benchmark(alg_two_trees($trias1, $trias2), 
                        setup=(shuffle!($trias2)), samples=10, evals=50, seconds=15)
        push!(times_two_trees, mean(b2).time * 1e-6)
    end
end

n = n1 .+ n2 
df = DataFrame(N_total=n, N1=n1, N2=n2, 
                one_tree_time_ms=times_one_tree, two_trees_time_ms=times_two_trees)

sort!(df)

filename = string(ARGS[1], "_", nthreads(), "threads_exec_time.csv")
CSV.write(filename, df)
