#
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using DifferentiableEigen
using Test

import DifferentiableEigen.ForwardDiffChainRules.ForwardDiff
import Zygote
import LinearAlgebra
import FiniteDifferences

A = [1.0 1.5 3.0; -0.3 -0.1 5.0; 1.1 3.4 -3.8] 
x = [1.0; 2.2; -1.2]

function eigvals(A)
    vals, vecs = eigen(A)
    return vals 
end

function eigvecs(A)
    vals, vecs = eigen(A)
    return vecs 
end

function to_diff_eigvals(x)
    A = x * x'
    return real(sum(eigvals(A)))
end

function to_diff_eigvecs(x)
    A = x * x'
    return real(sum(eigvecs(A)))
end

@testset "DifferentiableEigen.jl Tests" begin

    @testset "Results" begin
        v̂al, v̂ec = LinearAlgebra.eigen(A)
        val, vec = eigen(A)

        @test DifferentiableEigen.comp2Arr(v̂al) ≈ val
        @test DifferentiableEigen.comp2Arr(v̂ec) ≈ vec
    end

    grad_val_fd = nothing 
    grad_vec_fd = nothing

    @testset "FiniteDifferences" begin
        grad_val_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), to_diff_eigvals, x)[1]
        grad_vec_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), to_diff_eigvecs, x)[1]
    end

    @testset "ForwardDiff" begin
        grad_val = ForwardDiff.gradient(to_diff_eigvals, x)
        @test grad_val ≈ grad_val_fd

        grad_vec = ForwardDiff.gradient(to_diff_eigvecs, x)
        # vec_ratio = grad_vec ./ grad_vec_fd
        # for i in 2:length(vec_ratio)
        #     @test vec_ratio[i-1] ≈ vec_ratio[i]
        # end
    end

    # @testset "ReverseDiff" begin
    #     grad_val = ReverseDiff.gradient(to_diff_eigvals, x)
    #     @test grad_val ≈ grad_val_fd

    #     grad_vec = ReverseDiff.gradient(to_diff_eigvecs, x)
    #     @test grad_vec ≈ grad_vec_fd
    # end

    @testset "Zygote" begin
        grad_val = Zygote.gradient(to_diff_eigvals, x)[1]
        @test grad_val ≈ grad_val_fd

        grad_vec = Zygote.gradient(to_diff_eigvecs, x)[1]
        # vec_ratio = grad_vec ./ grad_vec_fd
        # for i in 2:length(vec_ratio)
        #     @test vec_ratio[i-1] ≈ vec_ratio[i]
        # end
    end
end
