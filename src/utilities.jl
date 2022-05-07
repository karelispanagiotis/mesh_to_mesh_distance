module Utilities

export select!,
       combine_boxes

using Meshes
using Base.Sort, Base.Order

function select!(v::AbstractVector, lo::Integer, hi::Integer, k::Integer, o::Ordering)
    @inbounds while lo < hi
        # range must be ≥ 3 to apply partitioning, hande range==2 seperately
        if hi-lo == 1
            if lt(o, v[hi], v[lo])
                v[lo], v[hi] = v[hi], v[lo]
            end
            return v[lo]
        end

        j = Base.Sort.partition!(v, lo, hi, o)

        if k < j
            hi = j - 1
        elseif k > j
            lo = j + 1
        else
            return v[j]
        end
    end
    return v[lo]
end

function select!(v::AbstractVector, 
                 lo::Integer,
                 hi::Integer,
                 k::Integer;
                 lt=isless,
                 by=identity,
                 rev::Union{Bool,Nothing}=nothing,
                 order::Ordering=Forward)
    select!(v, lo, hi, k, ord(lt, by, rev, order))
end

function combine_boxes(B1::Box, B2::Box)
    S = Point(min.(coordinates(B1.min), coordinates(B2.min)))
    T = Point(max.(coordinates(B1.max), coordinates(B2.max)))
    return Box(S, T)
end

end