module PrimitiveDistances

export distance²,
       distance_endpoints 

using Meshes, LinearAlgebra
using StaticArrays
using EnhancedGJK

#-------------------------------------------------------------------
# Segment to Segment distance²
#
# Returns the squared distance between a pair of segments.
# 
# The implementation follows the notation used in 
# an algorithm described in:
#
# Vladimir J. Lumesky
# On fast computation of distance between line segments.
# In Information Processing Letters, no. 21, pages 55-61, 1985
#-------------------------------------------------------------------
function distance²(AB::Segment{Dim,T}, CD::Segment{Dim,T}) where {Dim,T}
    A, B = vertices(AB)
    C, D = vertices(CD)

    d₁  = B - A
    d₂  = D - C
    d₁₂ = C - A 
 
    D₁ = d₁⋅d₁   
    D₂ = d₂⋅d₂
    R  = d₁⋅d₂

    S₁ = d₁⋅d₁₂
    S₂ = d₂⋅d₁₂
    

    if D₁≈0 && D₂≈0
        t = u = zero(T)      
    elseif D₁≈0
        t = zero(T)
        u = clamp(-S₂/D₂, 0, 1)
    elseif D₂≈0
        u = zero(T)
        t = clamp( S₁/D₁, 0, 1)
    elseif D₁*D₂ - R^2 ≈ 0
        t = zero(T)
        u = -S₂/D₂
        if u<0 || u>1
            u = clamp(u, 0, 1)
            t = clamp((u*R+S₁)/D₁, 0, 1)
        end
    else
        t = clamp((S₁*D₂ - S₂*R)/(D₁*D₂ - R^2), 0, 1)
        u = (t*R - S₂)/D₂
        if u<0 || u>1
            u = clamp(u, 0, 1)
            t = clamp((u*R+S₁)/D₁, 0, 1)
        end
    end

    DD = t*d₁ - u*d₂ - d₁₂
    return DD⋅DD
end

#-------------------------------------------------------------------
# vectices_to_triangle_check()
#
# Checks if the projection of P ∈ verts on the plane defined by
# triangle T, lies within T. If this is the case, the distance of 
# P to triangle T is equal to the distance of P to the plane itself.
# 
# The algorithm that checks if a projection of a point lies within
# the triangle T, is described in:
#
# Wolfgang Heidrich
# Computing the Barycentric Coordinates of a Projected Point
# In Journal of Graphics Tools, pages 9-12, 2005
#-------------------------------------------------------------------
function vectices_to_triangle_check(verts, t::Triangle{Dim,T}) where {Dim,T}
    P₁, P₂, P₃ = vertices(t)

    u = P₂ - P₁
    v = P₃ - P₁
    n  = u × v
    n² = n ⋅ n

    dist² = typemax(T)
    for P ∈ verts
        w = P - P₁

        #Compute Barycentric Coordinates of Projected Point
        γ = ( (u × w)⋅n )/n²  
        β = ( (w × v)⋅n )/n²
        α = 1 - γ - β

        if 0≤α≤1 && 0≤β≤1 && 0≤γ≤1
            dist² = min(dist², (n⋅w)^2/n²)
        end
    end

    return dist²
end

#-------------------------------------------------------------------
# Triangle to Triangle distance²()
#
# Returns the squared distance of two triangles
# 
# Checks all possible scenarios for the minimum distance:
#   * The minimum distance is between two edges or
#   * The minimum distance is between a vertex and a face or
#   * The minimum distance is zero, because the triangles intersect
#-------------------------------------------------------------------
function distance²(T1::Triangle{Dim,T}, T2::Triangle{Dim,T}) where {Dim,T}
    A, B, C = vertices(T1)
    X, Y, Z = vertices(T2)
    edges1 = (Segment(A,B), Segment(B,C), Segment(C,A))
    edges2 = (Segment(X,Y), Segment(Y,Z), Segment(Z,X))
    
    dist² = typemax(T)
    for seg₁ ∈ edges1
        for seg₂ ∈ edges2
            dist² = min(dist², distance²(seg₁, seg₂))
        end
    end
    
    dist² = min(dist², vectices_to_triangle_check(vertices(T1), T2))
    dist² = min(dist², vectices_to_triangle_check(vertices(T2), T1))
    return dist²
end

#-------------------------------------------------------------------
# AABB to AABB distance²()
#
# Returns the squared distance of two Axis-Aligned Bounding Boxes 
#-------------------------------------------------------------------
function distance²(B1::Box, B2::Box)
    v = max.(minimum(B1) - maximum(B2), 0)
    w = max.(minimum(B2) - maximum(B1), 0)
    return v⋅v + w⋅w 
end

#-------------------------------------------------------------------
# The same functions as above, but instead of distance, return 
# the endpoints of minimum distance between the primitives
#-------------------------------------------------------------------

function distance_endpoints(AB::Segment, CD::Segment)
    A, B = vertices(AB)
    C, D = vertices(CD)

    d₁  = B - A
    d₂  = D - C
    d₁₂ = C - A 
 
    D₁ = d₁⋅d₁   
    D₂ = d₂⋅d₂
    R  = d₁⋅d₂

    S₁ = d₁⋅d₁₂
    S₂ = d₂⋅d₁₂
    
    if D₁≈0 && D₂≈0
        t = u = 0.0f0      
    elseif D₁≈0
        t = 0.0f0
        u = clamp(-S₂/D₂, 0, 1)
    elseif D₂≈0
        u = 0.0f0
        t = clamp( S₁/D₁, 0, 1)
    elseif D₁*D₂ - R^2 ≈ 0.0
        t = 0.0f0
        u = -S₂/D₂
        if u<0 || u>1
            u = clamp(u, 0, 1)
            t = clamp((u*R+S₁)/D₁, 0, 1)
        end
    else
        t = clamp((S₁*D₂ - S₂*R)/(D₁*D₂ - R^2), 0, 1)
        u = (t*R - S₂)/D₂
        if u<0 || u>1
            u = clamp(u, 0, 1)
            t = clamp((u*R+S₁)/D₁, 0, 1)
        end
    end

    return Segment(AB(t), CD(u))
end

function vectices_to_triangle_endpoints(verts, T::Triangle)
    P₁, P₂, P₃ = vertices(T)

    u = P₂ - P₁
    v = P₃ - P₁
    n  = u × v
    n² = n ⋅ n

    endpoints = Segment(Point(-Inf32, -Inf32, -Inf32), Point(+Inf32, +Inf32, +Inf32))
    for P ∈ verts
        w = P - P₁

        #Compute Barycentric Coordinates of Projected Point
        γ = ( (u × w)⋅n )/n²  
        β = ( (w × v)⋅n )/n²
        α = 1 - γ - β

        if 0≤α≤1 && 0≤β≤1 && 0≤γ≤1 && (n⋅w)^2 / n² < measure(endpoints)
            endpoints = Segment(P, P - n * (n⋅w))
        end
    end

    return endpoints
end

function distance_endpoints(T1::Triangle, T2::Triangle)
    endpoints = Segment(Point3f(+Inf32, +Inf32, +Inf32), Point3f(-Inf32, -Inf32, -Inf32))
    for seg₁ ∈ segments(chains(T1)[1])
        for seg₂ ∈ segments(chains(T2)[1])
            tmp = distance_endpoints(seg₁, seg₂)
            if measure(tmp) < measure(endpoints) 
                endpoints = tmp
            end
        end
    end
    
    tmp = vectices_to_triangle_endpoints(vertices(T1), T2)
    if measure(tmp) < measure(endpoints) 
        endpoints = tmp
    end

    tmp = vectices_to_triangle_endpoints(vertices(T2), T1)
    if measure(tmp) < measure(endpoints) 
        endpoints = tmp
    end

    return endpoints
end

end