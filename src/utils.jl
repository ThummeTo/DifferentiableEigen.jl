#
# Copyright (c) 2023 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# makes Reals from ForwardDiff.Dual scalar/vector
function undual(e::AbstractArray)
    return collect(undual(c) for c in e)
end
function undual(e::Tuple)
    return (collect(undual(c) for c in e)...,)
end
function undual(e::ForwardDiff.Dual)
    return ForwardDiff.value(e)
end
function undual(::Nothing)
    return nothing
end
function undual(e)
    return e
end

# makes Reals from ForwardDiff/ReverseDiff.TrackedXXX scalar/vector
function unsense(e::AbstractArray)
    return collect(unsense(c) for c in e)
end
function unsense(e::Tuple)
    return (collect(unsense(c) for c in e)...,)
end
function unsense(e::ReverseDiff.TrackedReal)
    return ReverseDiff.value(e)
end
function unsense(e::ReverseDiff.TrackedArray)
    return ReverseDiff.value(e)
end
function unsense(e::ForwardDiff.Dual)
    return ForwardDiff.value(e)
end
function unsense(::Nothing)
    return nothing
end
function unsense(e)
    return e
end

# ToDo
function comp2Arr(comp::Union{AbstractVector,AbstractMatrix})
    l = length(comp)
    a = Real[]

    for i = 1:l
        re = real(comp[i])
        im = imag(comp[i])

        push!(a, re)
        push!(a, im)
    end

    return a
end

# ToDo
function arr2Comp(arr::AbstractVector, size)
    l = length(arr)
    a = Vector{Complex}()
    i = 1

    while i < l
        re = arr[i]
        im = arr[i+1]

        push!(a, Complex(re, im))

        # inc by 2
        i += 2
    end

    return reshape(a, size)
end
