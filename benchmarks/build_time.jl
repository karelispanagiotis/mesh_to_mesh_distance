include("../src/io.jl")
include("../src/skd_tree.jl")

using DataFrames
using CSV
using BenchmarkTools
using Meshes
using .MeshLoader
using .SKDTree

times = Vector{Float64}(undef, 0)
n_triangles = Vector{Int32}(undef, 0)
for (root, dirs, files) in walkdir("testcases/")
    for file in files 
        mesh = load_mesh(joinpath(root, file))
        triangles = collect(elements(mesh))
        b = @benchmark sKDTree($triangles)
        push!(times, mean(b).time*1e-6) #in milliseconds
        push!(n_triangles, length(collect(elements(mesh))))
    end
end

idx = sortperm(n_triangles)
n_triangles = n_triangles[idx]
times = times[idx]

df = DataFrame(N=n_triangles, time_ms=times)
filename = string(Threads.nthreads(), "threads_build_time.csv")
CSV.write(filename, df)

