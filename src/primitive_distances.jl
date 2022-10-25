module PrimitiveDistances

export  AABB_distance²,
        triangle_distance²,
        distance_endpoints, 
        PQP_SegPoints,
        PQP_TriDist

using Meshes, LinearAlgebra

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
function AABB_distance²(AB::Segment{Dim,T}, CD::Segment{Dim,T}) where {Dim,T}
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

function triangle_distance²(T1::Triangle{Dim,T}, T2::Triangle{Dim,T}) where {Dim,T}
    dist², = PQP_TriDist(T1, T2)
    return dist²
end

#-------------------------------------------------------------------
# AABB to AABB distance²()
#
# Returns the squared distance of two Axis-Aligned Bounding Boxes 
#-------------------------------------------------------------------
function AABB_distance²(B1::Box, B2::Box)
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

function PQP_SegPoints(s1::Segment, s2::Segment)
    return PQP_SegPoints(minimum(s1), maximum(s1) - minimum(s1),
                         minimum(s2), maximum(s2) - minimum(s2));
end

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


end