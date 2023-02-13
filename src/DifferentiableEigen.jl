#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module DifferentiableEigen

import LinearAlgebra: I, diag, rank, diagm
import LinearAlgebra

import ForwardDiffChainRules.ChainRulesCore: ZeroTangent, NoTangent
import ForwardDiffChainRules.ChainRulesCore
import ForwardDiffChainRules.ForwardDiff
import ForwardDiffChainRules

include(joinpath(@__DIR__, "utils.jl"))

# a wrapper to LinearAlgebra.eigen, but with real array output instead of complex array
function eigen(A::AbstractMatrix)
    A = undual(A)
    val, vec = LinearAlgebra.eigen(A)
    
    return comp2Arr(val), comp2Arr(vec) 
end
export eigen

function ChainRulesCore.frule((Δself, ΔA), ::typeof(eigen), A::AbstractMatrix)
    
    A = undual(A)
    
    eU = LinearAlgebra.eigen(A)
    e,U = eU
    n = size(A,1)

    Ω = comp2Arr(e), comp2Arr(U) 
    ∂e = ZeroTangent()
    ∂U = ZeroTangent()

    F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]

    UAU = ΔA

    if rank(U) == n
        UAU = inv(U) * ΔA * U
    end

    ∂e = diag(UAU) 
    ∂U = U * (F .* UAU)

    ∂Ω = (comp2Arr(∂e), comp2Arr(∂U)) 

    return Ω, ∂Ω 
end

function ChainRulesCore.rrule(::typeof(eigen), A::AbstractMatrix)
    
    eU = LinearAlgebra.eigen(A)

    e,U = eU
    n = size(A,1)

    Ω = eU

    function pullback(r̄)

        ē, Ū = r̄

        Ā = ZeroTangent()

        D̄ = nothing 
        
        if ē != nothing && ē != ZeroTangent()
            D̄ = diagm(ē)
        end
        
        if Ū === nothing
            Ā = inv(U)' * D̄ * U'

        elseif D̄ === nothing
            F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
            Ā = inv(U)'*(F .* (U' * Ū))*U'

        else
            F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
            Ā = inv(U)'*(D̄ + F .* (U' * Ū))*U'

        end

        f̄ = NoTangent()

        ∂Ω = f̄, Ā
    end

    return Ω, pullback 
end

import ForwardDiffChainRules: @ForwardDiff_frule
@ForwardDiff_frule eigen(A::AbstractMatrix{<:ForwardDiff.Dual})

end # module DifferentiableEigen
