module DistanceAlgorithms

export  alg_bruteforce,
        alg_bruteforce_bbox,
        alg_bruteforce_bbox_threads,
        alg_tree_queries

include("primitive_distances.jl")
include("skd_tree.jl")
using Meshes
using Base.Threads, .PrimitiveDistances, .SKDTree

function alg_bruteforce(trias1, trias2)
    mindist = Inf32
    tid1 = tid2 = typemax(Int)
    for i in 1:length(trias1)
        for j in 1:length(trias2)
            mindist, tid1, tid2 = min((mindist,tid1,tid2), (distance²(trias1[i], trias2[j]),i,j) ) 
        end
    end
    return mindist, tid1, tid2
end

function alg_bruteforce_bbox(trias1, trias2)
    boxes1 = boundingbox.(trias1)
    boxes2 = boundingbox.(trias2)

    mindist = Inf32
    tid1 = tid2 = typemax(Int)
    for i in 1:length(trias1)
        for j in 1:length(trias2)
            if distance²(boxes1[i], boxes2[j])<mindist
                mindist, tid1, tid2 = min( (mindist, tid1, tid2), (distance²(trias1[i], trias2[j]),i,j) ) 
            end
        end
    end
    return mindist, tid1, tid2
end

function alg_bruteforce_bbox_threads(trias1, trias2)
    getrange(N) = (work = ceil(Int, N/nthreads()); 1 + (threadid()-1)*work : min(threadid()*work, N))
    
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
                if distance²(boxes1[i], boxes2[j]) < l_mindist
                    l_mindist, l_tid1, l_tid2 = min( (l_mindist,l_tid1, l_tid2), (distance²(trias1[i],trias2[j]),i,j))
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
    tree = sKDTree(trias1)
    mindist, tid1, tid2 = Inf32, typemax(Int), typemax(Int)
    for i in 1:length(trias2)
        (mindist, tid1), tid2 = min( ((mindist,tid1),tid2), (nearest_neighbour(tree,trias2[i]; radius²=mindist),i) )
    end
    return mindist, tid1, tid2
end

end