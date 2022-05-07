module SKDTree

export sKDTree,
       nearest_neighbour

include("utilities.jl")
include("primitive_distances.jl")

using Meshes
using .Utilities, .PrimitiveDistances

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

    left, right  = v+1, v+2*(mid-lo+1)
    build_sKDTree(tree, left , lo   , mid,  next_split, centroids)
    build_sKDTree(tree, right, mid+1, hi ,  next_split, centroids)
    
    nodes[v] = combine_boxes(nodes[left], nodes[right])
    return nothing
end

function nn_search(tree::sKDTree{V,Dim,T}, 
                   v::Int,
                   lo::Int,
                   hi::Int,
                   query::Geometry{Dim, T}, 
                   query_aabb::Box{Dim, T},
                   radius²::T) where {V, Dim, T}
    data, nodes, indices, leaf_size = tree.data, tree.nodes, tree.indices, tree.leaf_size
    if lo == hi
        return distance²(query, data[indices[lo]]), indices[lo]
    end

    min_dist², nn_id = radius², typemax(eltype(indices))

    mid = (lo+hi)>>>1
    left, right  = v+1, v+2*(mid-lo+1)
    
    left_dist  = distance²(query_aabb, nodes[left])
    right_dist = distance²(query_aabb, nodes[right])
    
    if left_dist < right_dist
        # Search in the left sub-tree
        if left_dist < min_dist²
            min_dist², nn_id = min((min_dist², nn_id), nn_search(tree, left, lo, mid, query, query_aabb, min_dist²))
        end

        # Search in the right sub-tree
        if right_dist < min_dist²
            min_dist², nn_id = min((min_dist², nn_id), nn_search(tree, right, mid+1, hi, query, query_aabb, min_dist²))
        end 
    else
        # Search in the right sub-tree
        if right_dist < min_dist²
            min_dist², nn_id = min((min_dist², nn_id), nn_search(tree, right, mid+1, hi, query, query_aabb, min_dist²))
        end 

        # Search in the left sub-tree
        if left_dist < min_dist²
            min_dist², nn_id = min((min_dist², nn_id), nn_search(tree, left, lo, mid, query, query_aabb, min_dist²))
        end
    end

    return (min_dist², nn_id)
end

function nearest_neighbour(tree::sKDTree{V,Dim,T}, query::Geometry{Dim,T}; radius²::T=typemax(T)) where{V,Dim,T} 
    query_aabb = boundingbox(query)
    return nn_search(tree, 1, 1, length(tree.data), query, query_aabb, radius²)
end

function sKDTree(data::Vector{<:Geometry{Dim,T}}; leafsize=1) where {Dim,T}
    N = length(data)
    tree = sKDTree(data, Vector{Box{Dim,T}}(undef, 2*N), Vector(1:N), leafsize)
    centroids = centroid.(data)
    build_sKDTree(tree, 1, 1, N, 1, centroids)
    return tree
end

end

