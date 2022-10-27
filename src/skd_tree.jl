module SKDTree

export sKDTree,
       build_sKDTree,
       nearest_neighbour,
       nearest_neighbours

include("utilities.jl")
include("primitive_distances.jl")

using Meshes
using Base.Threads
using .Utilities
using .PrimitiveDistances

struct sKDTree{V, Dim, T}
    data::Vector{V}
    nodes::Vector{Box{Dim,T}}
    aabbs::Vector{Box{Dim,T}}
    indices::Vector{Int}
    leaf_size::Int
end

function build_sKDTree(tree::sKDTree,
                       v::Int,
                       lo::Int,
                       hi::Int,
                       depth::Int,
                       centroids::Vector{<:Point})
    data, nodes, indices, aabbs= tree.data, tree.nodes, tree.indices, tree.aabbs
    if hi-lo+1 ≤ tree.leaf_size
        nodes[v] = aabbs[indices[lo]]
        for i in lo+1:hi
            nodes[v] = combine_boxes(nodes[v], aabbs[indices[i]])
        end
        return nothing
    end

    mid = (lo+hi)>>>1
    split_dim::Int = mod(depth, 1:3) 
    select!(indices, lo, hi, mid, by = i -> coordinates(centroids[i])[split_dim])

    left, right  = v+1, v+2*(mid-lo+1)
    if nthreads() ≥ 2^(depth-1)
        # Build in parallel
        task = Threads.@spawn build_sKDTree(tree, left, lo, mid, depth+1, centroids)
        build_sKDTree(tree, right, mid+1, hi , depth+1, centroids)
        wait(task)
    else 
        # Build sequentially
        build_sKDTree(tree, left , lo   , mid, depth+1, centroids)
        build_sKDTree(tree, right, mid+1, hi , depth+1, centroids)
    end
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
    data, nodes, indices, aabbs= tree.data, tree.nodes, tree.indices, tree.aabbs
    min_dist², nn_id = radius², typemax(eltype(indices))
    
    if hi-lo+1 ≤ tree.leaf_size
        for i in lo:hi
            if AABB_distance²(query_aabb, aabbs[indices[i]]) < min_dist²
                min_dist², nn_id = min( (min_dist²,nn_id), (triangle_distance²(query, data[indices[i]]), indices[i]) )
            end
        end
        return min_dist², nn_id
    end

    mid = (lo+hi)>>>1
    left, right  = v+1, v+2*(mid-lo+1)
    
    left_dist  = AABB_distance²(query_aabb, nodes[left])
    right_dist = AABB_distance²(query_aabb, nodes[right])
    
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

function nns_search(tree1::sKDTree{V, Dim, T},
                    tree2::sKDTree{V, Dim, T},
                    v::Int,
                    lo::Int,
                    hi::Int,
                    radius²::T=typemax(T)) where {V, Dim, T}
    data, nodes, indices, aabbs= tree2.data, tree2.nodes, tree2.indices, tree2.aabbs
    min_dist², nn_id₁, nn_id₂  = radius², typemax(eltype(indices)), typemax(eltype(indices))

    if hi-lo+1 ≤ tree2.leaf_size
        for i in lo:hi
            tmp_min_dist², tmp_nn_id₁ = nn_search(tree1, 1, 1, length(tree1.data), data[indices[i]], aabbs[indices[i]], min_dist²)
            min_dist², nn_id₁, nn_id₂ = min((min_dist²,nn_id₁,nn_id₂), (tmp_min_dist²,tmp_nn_id₁,indices[i]))
        end
        return min_dist², nn_id₁, nn_id₂
    end

    mid = (lo+hi)>>>1
    left, right  = v+1, v+2*(mid-lo+1)

    left_dist  = AABB_distance²(nodes[left] , tree1.nodes[1])
    right_dist = AABB_distance²(nodes[right], tree1.nodes[1])

    if left_dist < right_dist
        # Search in the left sub-tree
        if left_dist < min_dist²
            min_dist², nn_id₁, nn_id₂ = min((min_dist²,nn_id₁,nn_id₂), nns_search(tree1, tree2, left, lo, mid, min_dist²))
        end

        # Search in the right sub-tree
        if right_dist < min_dist²
            min_dist², nn_id₁, nn_id₂ = min((min_dist²,nn_id₁,nn_id₂), nns_search(tree1, tree2, right, mid+1, hi, min_dist²))
        end 
    else
        # Search in the right sub-tree
        if right_dist < min_dist²
            min_dist², nn_id₁, nn_id₂ = min((min_dist²,nn_id₁,nn_id₂), nns_search(tree1, tree2, right, mid+1, hi, min_dist²))
        end 

        # Search in the left sub-tree
        if left_dist < min_dist²
            min_dist², nn_id₁, nn_id₂ = min((min_dist²,nn_id₁,nn_id₂), nns_search(tree1, tree2, left, lo, mid, min_dist²))
        end
    end

    return (min_dist², nn_id₁, nn_id₂)
end


function nearest_neighbour(tree::sKDTree{V,Dim,T}, 
                           query::Geometry{Dim,T};
                           radius²::T=typemax(T)) where{V,Dim,T} 
    return nn_search(tree, 1, 1, length(tree.data), query, boundingbox(query), radius²)
end

function nearest_neighbours(tree1::sKDTree{V,Dim,T}, 
                           tree2::sKDTree{V,Dim,T}) where{V,Dim,T} 
    return nns_search(tree1, tree2, 1, 1, length(tree2.data)) 
end

function sKDTree(data::Vector{<:Geometry{Dim,T}}; leafsize::Int=ceil(Int,log2(length(data)))) where {Dim,T}
    n = length(data)
    tree = sKDTree(data, Vector{Box{Dim,T}}(undef, 2*n), boundingbox.(data), Vector(1:n), leafsize)
    build_sKDTree(tree, 1, 1, n, 1, center.(tree.aabbs))
    return tree
end

end

