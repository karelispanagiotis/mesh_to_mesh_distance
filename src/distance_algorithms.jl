module DistanceAlgorithms

export  alg_bruteforce,
        alg_bruteforce_bbox,
        alg_bruteforce_bbox_threads,
        alg_tree_queries,
        alg_two_trees

include("primitive_distances.jl")
include("skd_tree.jl")
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

function alg_bruteforce(trias1, trias2)
    mindist = typemax(coordtype(eltype(trias1)))
    tid1 = tid2 = typemax(Int)
    for i in 1:length(trias1)
        for j in 1:length(trias2)
            mindist, tid1, tid2 = min((mindist,tid1,tid2), (triangle_distance²(trias1[i], trias2[j]),i,j) ) 
        end
    end
    return mindist, tid1, tid2
end

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

function alg_bruteforce_bbox_threads(trias1, trias2)
    boxes1 = boundingbox.(trias1)
    boxes2 = boundingbox.(trias2)

    mindist = zeros(Float32, nthreads())
    tid1 = zeros(Int, nthreads()) 
    tid2 = zeros(Int, nthreads())
    @threads for i in 1:nthreads()
        l_mindist = Inf32
        l_tid1 = l_tid2 = typemax(Int)
        for i in getrange(length(trias1))
            for j in 1:length(trias2)
                if AABB_distance²(boxes1[i], boxes2[j]) < l_mindist
                    l_mindist, l_tid1, l_tid2 = min( (l_mindist,l_tid1, l_tid2), (triangle_distance²(trias1[i],trias2[j]),i,j))
                end
            end
        end
        mindist[threadid()] = l_mindist
        tid1[threadid()] = l_tid1
        tid2[threadid()] = l_tid2
    end
    i = argmin(mindist)
    return mindist[i], tid1[i], tid2[i]
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
    task = @spawn sKDTree(trias1)
    tree2 = sKDTree(trias2)
    tree1 = fetch(task)
    return nearest_neighbours(tree1, tree2)
end

end