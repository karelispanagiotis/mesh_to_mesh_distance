using Meshes, LinearAlgebra

#--------------------------------------------------------------
# segment_to_segment_dist2()
#
# Returns the squared distance between a pair of segments.
# 
# The algorithm is described in:
#
# Vladimir J. Lumesky
# On fast computation of distance between line segments.
# In Information Processing Letters, no. 21, pages 55-61, 1985
#--------------------------------------------------------------
function segment_to_segment_dist2(A::Point, B::Point, C::Point, D::Point)
    d1  = B - A
    d2  = D - C
    d12 = C - A

    D1 = dot(d1, d1)    
    D2 = dot(d2, d2)
    R  = dot(d1, d2)

    S1 = dot(d1, d12)
    S2 = dot(d2, d12)
    
    if D1≈0 && D2≈0
        t = u = 0.0f0      
    elseif D1≈0
        t = 0.0f
        u = clamp(-S2/D2, 0, 1)
    elseif D2≈0
        u = 0.0f0
        t = clamp( S1/D1, 0, 1)
    elseif D1*D2 - R^2 ≈ 0.0
        t = 0.0f0
        u = -S2/D2
        if u<0 || u>1
            u = clamp(u, 0, 1)
            t = clamp((u*R+S1)/D1, 0, 1)
        end
    else
        t = clamp((S1*D2 - S2*R)/(D1*D2 - R^2), 0, 1)
        u = (t*R - S2)/D2
        if u<0 || u>1
            u = clamp(u, 0, 1)
            t = clamp((u*R+S1)/D1, 0, 1)
        end
    end

    DD = t*d1 - u*d2 - d12
    return dot(DD, DD)
end