include("../src/utilities.jl")
include("../src/io.jl")

using Meshes
using Base.Threads
using LinearAlgebra
using Random
using .Utilities
using .MeshLoader

# distance_algorithms.jl #
function alg_bruteforce_bbox(trias1, trias2)
    boxes1 = boundingbox.(trias1)
    boxes2 = boundingbox.(trias2)

    mindist = typemax(coordtype(eltype(trias1)))
    tid1 = tid2 = typemax(Int)
    for i in 1:length(trias1)
        for j in 1:length(trias2)
            if AABB_distance²(boxes1[i], boxes2[j])<mindist
                mindist, tid1, tid2 = min( (mindist, tid1, tid2), (triangle_distance²(trias1[i], trias2[j]),i,j) ) 
            end
        end
    end
    return mindist, tid1, tid2
end

function alg_tree_queries(trias1, trias2)
    if length(trias1) < length(trias2)
        trias1, trias2 = trias2, trias1
    end
    tree = sKDTree(trias1)

    mindist = typemax(coordtype(eltype(trias1)))
    tid1 = tid2 = typemax(Int)
    for j in eachindex(trias2)  
        tmp_dist, tmp_i = nearest_neighbour(tree, trias2[j]; radius²=mindist)
        mindist,tid1,tid2 = min((mindist,tid1,tid2), (tmp_dist,tmp_i,j))
    end
    return mindist, tid1, tid2 
end

function alg_two_trees(trias1, trias2)
    task = Threads.@spawn sKDTree(trias1)
    tree2 = sKDTree(trias2)
    tree1 = fetch(task)
    return nearest_neighbours(tree1, tree2)
end

# primitive_distances.jl #
function PQP_SegPoints(P::Point{Dim,Type}, A::Vec{Dim,Type},
                       Q::Point{Dim,Type}, B::Vec{Dim,Type} ) where {Dim,Type}
    T = Q - P
    A_dot_A = A ⋅ A 
    B_dot_B = B ⋅ B
    A_dot_B = A ⋅ B 
    A_dot_T = A ⋅ T
    B_dot_T = B ⋅ T;

    denom = A_dot_A*B_dot_B - A_dot_B*A_dot_B
    
    t = (A_dot_T*B_dot_B - B_dot_T*A_dot_B) / denom
    t = isnan(t) ? 0 : clamp(t, 0, 1)
    
    u = (t*A_dot_B - B_dot_T) / B_dot_B

    if u≤0 || isnan(u)
        Y = Q 
        t = A_dot_T / A_dot_A
        if t≤0 || isnan(t)
            X = P 
            VEC = Q - P
        elseif t≥1 
            X  = P + A 
            VEC = Q - X 
        else 
            X = P + t*A
            VEC = A × (T × A)
        end
    elseif u≥1 
        Y = Q + B 
        t = (A_dot_B + A_dot_T) / A_dot_A
        if t≤0 || isnan(t) 
            X = P 
            VEC = Y - P
        elseif t≥1
            X = P + A 
            VEC = Y - X 
        else
            X = P + t*A 
            T = Y - P 
            VEC = A × (T × A)
        end 
    else 
        Y = Q + u*B
        if t≤0 || isnan(t)
            X = P 
            VEC = B × (T × B)
        elseif t≥1
            X = P + A 
            T = Q - X 
            VEC = B × (T × B) 
        else
            X = P + t*A 
            VEC = A × B
            if VEC⋅VEC ≈ 0 
                VEC = Y - X
            elseif VEC ⋅ T < 0 
                VEC = -VEC 
            end
        end
    end
    return VEC, X, Y
end

function PQP_TriDist(t1::Triangle{Dim,Type}, t2::Triangle{Dim,Type}) where {Dim, Type}
    S = vertices(t1)
    T = vertices(t2)
    Sv = (S[2] - S[1], S[3] - S[2], S[1] - S[3])
    Tv = (T[2] - T[1], T[3] - T[2], T[1] - T[3])

    minP = S[1]
    minQ = S[1]
    shown_disjoint = false 
    mindd = typemax(Type)
    for i in 1:3 
        for j in 1:3 
            VEC, P, Q = PQP_SegPoints(S[i], Sv[i], T[j], Tv[j])
            V = Q - P
            dd = V⋅V
            
            if dd ≤ mindd
                minP = P 
                minQ = Q 
                mindd = dd 

                Z = S[mod(i+2,1:3)] - P 
                a = Z ⋅ VEC 
                Z = T[mod(j+2,1:3)] - Q 
                b = Z ⋅ VEC 

                if a≤0 && b≥0
                    return dd, P, Q 
                end 

                p = V⋅VEC 
                a = max(a, 0)
                b = min(b, 0)
                if p-a+b > 0
                    shown_disjoint = true
                end
            end
        end
    end

    Sn = Sv[1] × Sv[2]
    Snl = Sn⋅Sn

    if Snl > 1e-15 
        Vd = (S[1]-T[1], S[1]-T[2], S[1]-T[3]) 
        Tp = (Vd[1]⋅Sn, Vd[2]⋅Sn, Vd[3]⋅Sn)

        point = -1 
        if Tp[1] > 0 && Tp[2] > 0 && Tp[3] > 0
            point = argmin(Tp)
        elseif Tp[1] < 0 && Tp[2] < 0 && Tp[3] < 0
            point = argmax(Tp)
        end

        if point ≥ 1
            shown_disjoint = true 

            V = T[point] - S[1]
            Z = Sn × Sv[1]
            if V⋅Z > 0 
                V = T[point] - S[2]
                Z = Sn × Sv[2]
                if V⋅Z > 0
                    V = T[point] - S[3]
                    Z = Sn × Sv[3]
                    if V⋅Z > 0
                        P = T[point] + Tp[point]/Snl * Sn 
                        Q = T[point]
                        V = P-Q
                        return V⋅V, P, Q
                    end
                end
            end
        end
    end

    Tn = Tv[1] × Tv[2]
    Tnl = Tn⋅Tn 

    if Tnl > 1e-15
        Vd = (T[1]-S[1], T[1]-S[2], T[1]-S[3])
        Sp = (Vd[1]⋅Tn, Vd[2]⋅Tn, Vd[3]⋅Tn)

        point = -1
        if Sp[1] > 0 && Sp[2] > 0 && Sp[3] > 0
            point = argmin(Sp)
        elseif Sp[1] < 0 && Sp[2] < 0 && Sp[3] < 0
            point = argmax(Sp)
        end

        if point ≥ 1
            shown_disjoint = true

            V = S[point] - T[1]
            Z = Tn × Tv[1]
            if V⋅Z > 0 
                V = S[point] - T[2]
                Z = Tn × Tv[2]
                if V⋅Z > 0
                    V = S[point] - T[3]
                    Z = Tn × Tv[3]
                    if V⋅Z > 0 
                        P = S[point]
                        Q = S[point] + Sp[point]/Tnl * Tn
                        V = P-Q
                        return V⋅V, P, Q 
                    end
                end
            end
        end
    end

    if shown_disjoint == true
        P = minP
        Q = minQ 
        V = P-Q
        return V⋅V, P, Q 
    end
    return zero(Type), P, Q
end

function triangle_distance²(T1::Triangle{Dim,T}, T2::Triangle{Dim,T}) where {Dim,T}
    tria_count[] += 1
    dist², = PQP_TriDist(T1, T2)
    return dist²
end

function AABB_distance²(B1::Box, B2::Box)
    aabb_count[] += 1
    v = max.(minimum(B1) - maximum(B2), 0)
    w = max.(minimum(B2) - maximum(B1), 0)
    return v⋅v + w⋅w 
end

# skd_tree.jl #
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
    Utilities.select!(indices, lo, hi, mid, by = i -> coordinates(centroids[i])[split_dim])

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
# SCRIPT BEGINS HERE #
using CoordinateTransformations
using Random
using CSV
using DataFrames
using StaticArrays

function apply_tranform(transform::Transformation, obj::Mesh)
    new_points = transform.(vertices(obj))
    return SimpleMesh(new_points, topology(obj))
end

global const aabb_count = Ref{Int64}(0)
global const tria_count = Ref{Int64}(0)
const triangle_cost = 385.7
const aabb_cost = 23.5


mesh1 = load_mesh("testcases/airplanes/obj1_topairplane_highres.obj")
mesh2 = load_mesh("testcases/airplanes/obj2_bottomairplane_highres.obj")

# dir = SVector(0,1,0)
dir = SVector(0,0,5)

mesh2_aabb = boundingbox(mesh2)
const normalization_constant = norm(maximum(mesh2_aabb) - minimum(mesh2_aabb))

dist = 0.0
costs_one_tree = Vector{Float64}(undef, 0)
costs_two_trees = Vector{Float64}(undef, 0)
normal_distances = Vector{Float64}(undef, 0)
while(dist/normalization_constant < 1000.0)
    
    trias1 = collect(elements(mesh1))
    trias2 = collect(elements(mesh2))

    shuffle!(trias1)
    shuffle!(trias2)

    aabb_count[] = tria_count[] = 0    
    alg_tree_queries(trias2, trias1)
    one_tree_cost = aabb_count[]*aabb_cost + tria_count[] * triangle_cost

    aabb_count[] = tria_count[] = 0    
    alg_two_trees(trias2, trias1)
    two_trees_cost = aabb_count[]*aabb_cost + tria_count[] * triangle_cost

    global dist, = alg_tree_queries(trias1, trias2)
    dist = sqrt(dist)

    push!(normal_distances, dist/normalization_constant)
    push!(costs_one_tree, one_tree_cost)
    push!(costs_two_trees, two_trees_cost)
    global dir *= 1.1
    global mesh2 = apply_tranform(Translation(dir), mesh2)
end

df = DataFrame(NormDist=normal_distances, 
                cost_one_tree=costs_one_tree, cost_two_trees=costs_two_trees)

CSV.write("seach_cost_against_distance.csv", df)