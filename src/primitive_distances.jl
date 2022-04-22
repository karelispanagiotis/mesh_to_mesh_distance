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
        t = 0.0f
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

    DD = t*d₁ - u*d₂ - d₁₂
    return DD⋅DD
end