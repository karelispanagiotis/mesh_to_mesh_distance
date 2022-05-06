module SKDTree

export sKDTree, build_sKDTree

include("utilities.jl")
using Meshes
using .Utilities

struct sKDTree{V, Dim, T}
    data::Vector{V}
    nodes::Vector{Box{Dim,T}}
    indices::Vector{Int}
    leaf_size::Int
end

function build_sKDTree(tree::sKDTree,
                       v::Int,
                       lo::Int,
                       hi::Int,
                       split_dim::Int,
                       centroids::Vector{<:Point})
    data, nodes, indices, leaf_size = tree.data, tree.nodes, tree.indices, tree.leaf_size
    if hi == lo
        nodes[v] = boundingbox(data[indices[lo]])
        return nothing
    end

    mid = (lo+hi)>>>1
    select!(indices, lo, hi, mid, by = i -> coordinates(centroids[i])[split_dim])

    next_split::Int = mod(split_dim+1, 1:3)

    left  = v+1
    right = v+2*(mid-lo+1)
    build_sKDTree(tree, left , lo   , mid,  next_split, centroids)
    build_sKDTree(tree, right, mid+1, hi ,  next_split, centroids)
    
    nodes[v] = combine_boxes(nodes[left], nodes[right])
    return nothing
end

function sKDTree(data::Vector{<:Geometry{Dim,T}}; leafsize=1) where {Dim,T}
    N = length(data)
    tree = sKDTree(data, Vector{Box{Dim,T}}(undef, 2*N), Vector(1:N), leafsize)
    centroids = centroid.(data)
    build_sKDTree(tree, 1, 1, N, 1, centroids)
    return tree
end

end

