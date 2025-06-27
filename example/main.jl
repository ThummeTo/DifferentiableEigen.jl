#
# Copyright (c) 2023 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using DiffEqFlux, Lux
using Optimization, OptimizationOptimisers, ComponentArrays
#import SciMLBase: RightRootFind
using DifferentialEquations
using DiffEqCallbacks
using SciMLBase: ReturnCode
using DiffEqFlux: ODEFunction, basic_tgrad, ODEProblem, ZygoteVJP, InterpolatingAdjoint, solve, NeuralODE
using Plots
import ForwardDiff
#import LinearAlgebra: eigvals, eigvecs, det, UpperTriangular, I, Diagonal, rank, norm
#import LinearAlgebra
#import FMIBase: undual, isdual
#using Printf: @sprintf
using LaTeXStrings
#using NLsolve
using Colors
using ChainRulesCore
using DifferentiableEigen
using Random 
using JLD2
using Statistics, StatsBase

const rng = Random.default_rng()
#           ["SOL", "FRQ", "DMP", "STB", "OSC", "STF"]
#gradScale= [  1e0,  1e1,   1e0,   1e3,   1e1,  1e-1] 
#gradScale= [  1e0,  1e1,   1e1,   1e2,   1e1,  1e-1] 
#gradScale= [  1e0,  1e1,   1e1,   1e1,   1e1,  1e-2]
#gradScale= [  1e0,  1e1,   1e1,  1e-1,   1e1,  1e-3] # todo: make const
gradScale = [  1e0,  1e1,   1e1,  1e-1,   1e1,  1e-4] # todo: make const
const adtype = Optimization.AutoForwardDiff(;chunksize=32)

# setup an almost neutral neural network (so we still can see the original system, even if we have some ANN added)
const layerwidth = 32
global v1, v2
function preprocess(x)
    global v1, v2

    v1 = x[2]
    if numStates == 4
        v2 = x[4]
    end

    return x .* [posScale, velScale]
end
function postprocess(a)
    global v1, v2

    if numStates == 2
        return [v1, a[1] / accScale]
    else
        return [v1, a[1] / accScale, v2, a[2] / accScale]
    end
end

const _eigen = DifferentiableEigen.eigen
function mae(a,b)
    len_a = length(a)
    @assert len_a == length(b)
    return sum(abs.(a .- b)) / len_a
end

function prepareEigPairs(eigvalData::AbstractVector{<:AbstractVector}; eps=1e-8)
    eigsPairs = Vector{Vector{Int}}[]
    for eigs in eigvalData
        pairs = prepareEigPairs(eigs)
       
        push!(eigsPairs, pairs)
        
    end
    return eigsPairs
end

function prepareEigPairs(eigs::AbstractVector{<:Real}; eps=1e-8)
    
    i=1
    pairs = Vector{Int}[]
    while i <= (length(eigs)-2)
        if abs(eigs[i] - eigs[i+2]) < eps
            push!(pairs, [Int((i-1)/2+1), Int((i+1)/2+1)])
        end
        i += 2
    end
    
    return pairs
end

function prepareStiffness(eigvalData)
    stiffnessData = collect(getStiffness(eigs) for eigs in eigvalData)
    return stiffnessData
end

function startPlot(args...; kwargs...)
    return Plots.plot(args...; background_color_legend=colorant"rgba(255,255,255,0.7)", kwargs...)
end

# right side of the ODE (out-of-place)
function f(u, p, t)
    return neuralODE(u, p)[1]
end 

function translational_pendulum(x, p)
    
    c, d = p
    
    s = x[1]
    v = x[2]
    a = -c*s -d*v

    return [v, a]
end 

function vanderpol(_x, p)
    
    μ  = p[1]

    x = _x[1]
    dx = _x[2]
    ddx = μ * (1- x^2)*dx - x

    return [dx, ddx]
end 

function translational_doublependulum(x)
    
    c1 = 10.0
    d1 = 0.1
    c2 = 25.0
    d2 = 0.1

    s1 = x[1]
    v1 = x[2]
    s2 = x[3]
    v2 = x[4]
    a1 =  c2*(s2-s1) +d2*(v2-v1) -c1*(s1) -d1*v1
    a2 = -c2*(s2-s1) -d2*(v2-v1)

    return [v1, a1, v2, a2]
end 

function rotational_doublependulum_loop!(res, iter, x)
    m1 = 1.0 
    m2 = 1.0
    l1 = 0.3 
    l2 = 0.3
    g = 9.81
    
    α1 = x[1]
    dα1 = x[2]
    α2 = x[3]
    dα2 = x[4]

    ddα1 = iter[1]
    ddα2 = iter[2]

    c = cos(α1-α2)
    s = sin(α1-α2)

    res[1] = (m2*g*sin(α2)*c - m2*s*(l1*dα1^2*c + L2*dα2^2) - (m1+m2)*g*sin(α1)) / l1 / (m1 + m2*s^2)
    res[2] = ((m1+m2)*(l1*dα1^2*s - g*sin(α2) + g*sin(α1)*c) + m2*l2*dα2^2*s*c) / l2 / (m1 + m2*s^2)

    nothing
end

global lastResiduum = zeros(2)
function rotational_doublependulum(x, p)

    global lastResiduum

    m1, m2, l1, l2, g = p
    
    α1 = x[1]
    dα1 = x[2]
    α2 = x[3]
    dα2 = x[4]

    #ddα1 =  -(m2*l2*ddα2*cos(α1-α2)) / ((m1+m2) * l1)
    #ddα2 =  -(m2*l1*ddα1*cos(α1-α2) - m2*l1*dα1*dα1*sin(α1-α2) + m2*g*sin(α2)) / (m2*l2)

    # if lastResiduum == nothing
    #     lastResiduum = copy(x)
    # end

    # res = nlsolve((res, iter) -> rotational_doublependulum_loop!(res, iter, x), lastResiduum)

    # ddα1 = res.zero[1]
    # ddα2 = res.zero[2]

    c = cos(α1-α2)
    s = sin(α1-α2)

    ddα1 = (m2*g*sin(α2)*c - m2*s*(l1*dα1^2*c + l2*dα2^2) - (m1+m2)*g*sin(α1)) / l1 / (m1 + m2*s^2)
    ddα2 = ((m1+m2)*(l1*dα1^2*s - g*sin(α2) + g*sin(α1)*c) + m2*l2*dα2^2*s*c) / l2 / (m1 + m2*s^2)

    #lastResiduum = copy(res.zero)

    return [dα1, ddα1, dα2, ddα2]
end 

function rotational_pendulum(x, p)

    global lastResiduum

    l, g = p
    
    α = x[1]
    dα = x[2]
    ddα = -g/l*sin(α)

    return [dα, ddα]
end 

function duffing(_x, p)

    global lastResiduum

    α, β, γ = p
    
    x = _x[1]
    dx = _x[2]

    ddx = -γ * dx - α * x - β * (x^3)

    return [dx, ddx]
end 

### ML

# function _eigen(A::AbstractMatrix)
#     A = undual(A)
#     val, vec = LinearAlgebra.eigen(A; sortby=LinearAlgebra.eigsortby)
    
#     @assert !isdual(val) "!!!"
#     @assert !isdual(vec) "!!!"

#     return comp2Arr(val), comp2Arr(vec) # return real.(val), real.(vec)
# end

# function ChainRulesCore.frule((Δself, ΔA), ::typeof(_eigen), A::AbstractMatrix)
#     #@info "frule start"

#     A = undual(A)
    
#     eV = LinearAlgebra.eigen(A; sortby=LinearAlgebra.eigsortby)
#     e,V = eV
#     n = size(A,1)

#     Ω = comp2Arr(e), comp2Arr(V) # Ω = real.(e), real.(V)
#     ∂e = ZeroTangent()
#     ∂V = ZeroTangent()

#     #D = Diagonal(e)
#     U = V
#     F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]

#     @assert size(U) == size(ΔA)

#     UAU = ΔA

#     if rank(U) == n
#         UAU = inv(U) * ΔA * U
#     end

#     ∂e = LinearAlgebra.diag(UAU) # exact: I .* UAU
#     ∂V = U * (F .* UAU)

#     ∂Ω = (comp2Arr(∂e), comp2Arr(∂V))  # (real.(∂e), real.(∂V)) 

#     #@info "frule end |∂e| = $(sum(∂e)), |∂V| = $(sum(∂V))"

#     @assert !isdual(∂e) "!!!"
#     @assert !isdual(∂V) "!!!"
#     @assert !isdual(Ω[1]) "!!!"
#     @assert !isdual(Ω[2]) "!!!"

#     return Ω, ∂Ω 
# end

# function ChainRulesCore.rrule(::typeof(_eigen), A::AbstractMatrix)
    
#     eU = eigen(A)
#     e,U = eU
#     n = size(A,1)

#     Ω = eU

#     function pullback(r̄)

#         ē, Ū = r̄

#         Ā = ZeroTangent()

#         D̄ = nothing 
        
#         if ē != nothing
#             D̄ = Diagonal(ē)
#         end
        
#         if Ū === nothing
#             Ā = inv(U)' * D̄ * U'

#         elseif D === nothing
#             F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
#             Ā = inv(U)'*(F .* (U' * Ū))*U'

#         else
#             F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
#             Ā = inv(U)'*(D̄ + F .* (U' * Ū))*U'

#         end

#         f̄ = NoTangent()

#         ∂Ω = f̄, Ā
#     end

#     return Ω, pullback 
# end

# import ForwardDiffChainRules: @ForwardDiff_frule
# @ForwardDiff_frule _eigen(A::AbstractMatrix{<:ForwardDiff.Dual})

function comp2Arr(comp)
    l = length(comp)
    a = Real[]
    for i in 1:l
        re = real(comp[i])
        im = imag(comp[i])

        push!(a, re)
        push!(a, im)
    end

    return a
end

function arr2Comp(arr, size)
    l = length(arr)
    a = Vector{Complex}()
    i = 1
    while i < l
        re = arr[i]
        im = arr[i+1]

        push!(a, Complex(re, im))
        
        # inc by 2
        i+=2
    end

    ret =  reshape(a, size)

    return ret
end

# global eigsRe = zeros(numStates)
# global eigsIm = zeros(numStates)

# function trackEigvals(eigs)
#     global eigsRe, eigsIm

#     numEigs = Int(length(eigs)/2)

#     tracked = zeros(Bool, numEigs)
#     order = []
    
#     for i in 1:numEigs

#         re_i = eigsRe[i] 
#         im_i = eigsIm[i]

#         closest = 0 
#         closestDist = Inf

#         for j in 1:numEigs
#             if !tracked[j]

#                 re_j = eigs[(j-1)*2+1]
#                 im_j = eigs[j*2]
#                 dist = sqrt((re_i-re_j)^2 + (im_i-im_j)^2)

#                 if dist < closestDist
#                     closest = j
#                     closestDist = dist
#                 end

#             end
#         end

#         tracked[closest] = true
#         push!(order, closest)

#         eigsRe[i] = eigs[(closest-1)*2+1]
#         eigsIm[i] = eigs[closest*2]
#     end

#     return order
# end

function loss_test(neuralODE, u0_test, p, st)
    solution = neuralODE(u0_test, p, st)[1]

    # generate data for new u0_test
    refProb = ODEProblem{false}(refODE, u0_test, tspan);
    refSol = solve(refProb; saveat=saveat)
    posData = collect(u[1] for u in refSol.u) 

    numTs = length(solution.t)
    lossSolution = gradSolution(solution; 
        posData=posData,
        error=(x, y)->abs.(x-y) / numStates / numTs )

    return lossSolution
end

function loss_parts(neuralODE, u0, p, st, gradFilter)

    #solution = solve(prob, neuralODE.args...; saveat=saveat, neuralODE.kwargs..., p=p) # sensealg=sensealg, 
    solution = neuralODE(u0, p, st)[1]
    
    _f = _u -> neuralODE.model(_u, p, st)[1]

    #global eigsArray
    eigsArray = []

    # if some eigenvalues are needed
    if sum(gradFilter[2:end]) > 0
        # eigenvalues (and sensitivities) are only computed a single time
        eigsArray = collect(_eigen(ForwardDiff.jacobian(_f, x))[1] for x in solution.u)
    end

    numTs = length(solution.t)
    numEigs = length(eigsArray)/2

    lossSolution = 0.0
    lossFrequency = 0.0
    lossDamping = 0.0
    lossStability = 0.0
    lossOscillation = 0.0
    lossStiffness = 0.0

    if gradFilter[1] == 1
        lossSolution = gradSolution(solution; error=(x, y)->abs.(x-y) / numStates / numTs )
    end

    if gradFilter[2] == 1
        lossFrequency = gradFrequency(eigsArray, eigvalData; error=(x, y)->abs(x-y) / numEigs /numTs )
    end
    
    if gradFilter[3] == 1
        lossDamping = gradDamping(eigsArray, eigvalData; error=(x, y)->abs(x-y) / numEigs /numTs)
    end
    
    if gradFilter[4] == 1
        lossStability = gradStability(eigsArray; error= (x) -> max(x, 0.0) / numEigs / numTs)
    end
    
    if gradFilter[5] == 1
        lossOscillation = gradOscillation(eigsArray, eigvalData; error=(x, y)->abs(x-y) / numEigs /numTs) # , eigsPairs
    end 
    
    if gradFilter[6] == 1 
        lossStiffness = gradStiffness(eigsArray, stiffnessData; error=(x, y)->abs(x-y) /numTs )
    end 

    return [lossSolution, lossFrequency, lossDamping, lossStability, lossOscillation, lossStiffness]
end

function loss(neuralODE, u0, p, st, loss_parts, gradFilter, gradScale)
    loss = 0.0 

    parts = loss_parts(neuralODE, u0, p, st, gradFilter)

    for i in 1:length(parts)
        # we can't train on Inf!
        if !isinf(parts[i])
            loss += parts[i] * gradScale[i]
        end
    end

    return loss
end

function gradSolution(solution; error=mae, posData=posData)

    if solution.retcode != ReturnCode.Success
        return Inf 
    end

    posNet = collect(data[1] for data in solution.u) 
    #velNet = collect(data[2] for data in solution.u)

    pos2Net = nothing 
    #vel2Net = nothing
    if numStates == 4
        pos2Net = collect(data[3] for data in solution.u) 
        #vel2Net = collect(data[4] for data in solution.u)
    end

    lossSolution = posScale * sum(collect(error(posNet[i], posData[i]) for i in 1:length(solution.t)))

    if numStates == 4 
        lossSolution += posScale * sum(collect(error(pos2Net[i], pos2Data[i]) for i in 1:length(solution.t)))
    end

    return lossSolution
end

function getFrequency(re, im)
    #return sqrt(re^2 + im^2)
    return abs(im) / (2.0 * π) 
end

function getDamping(re, im; eps=1e-8)
    #return cos(atan(im, re))
    len = sqrt(re^2 + im^2)
    if len < eps
        return 0.0
    else
        return -re / len
    end 
end

function getStiffness(eigs, eps=1e-8)
    numEigs = Int(length(eigs)/2)

    _min = Inf
    _max = -Inf
    for j in 1:numEigs
        re = abs(eigs[(j-1)*2+1])

        if re > _max 
            _max = re 
        end

        if re < _min 
            _min = re 
        end
    end

    if _min == _max 
        return 1.0
    elseif _min > eps 
        return _max/_min
    else 
        return Inf
    end
end

function gradEigCompare(func, eigsArray, eigsArrayData; error=mae)
    loss = 0.0

    # for every time step i
    for i in 1:length(eigsArray) 

        eigs = eigsArray[i]
        eigsData = eigsArrayData[i]

        numEigs = Int(length(eigs)/2)

        pairs = prepareEigPairs(eigs)
        matchedPairs = zeros(Bool, length(pairs))
        pairsData = prepareEigPairs(eigsData)

        # for every pair in DATA
        for j in 1:length(pairsData)

            minError = Inf
            minErrorIndex = nothing 

            pairData = pairsData[j]
            î1 = pairData[1]
            î2 = pairData[2]

            r̂e1 = eigsData[(î1-1)*2+1]
            îm1 = eigsData[î1*2]
            r̂e2 = eigsData[(î2-1)*2+1]
            îm2 = eigsData[î2*2]
            
            # for every pair in NODE
            for k in 1:length(pairs)

                # if unmatched pair
                if matchedPairs[k]
                    continue
                end

                pair = pairs[k]
                i1 = pair[1]
                i2 = pair[2]

                re1 = eigs[(i1-1)*2+1]
                im1 = eigs[i1*2]
                re2 = eigs[(i2-1)*2+1]
                im2 = eigs[i2*2]

                er = error(func(re1, im1), func(r̂e1, îm1)) + error(func(re2, im2), func(r̂e2, îm2))
                if er < minError
                    minError = er 
                    minErrorIndex = k 
                end
            end

            if minErrorIndex != nothing
                loss += minError
                matchedPairs[minErrorIndex] = true
            end

        end

        # for j in 1:numEigs
        #     re = eigs[(j-1)*2+1]
        #     im = eigs[j*2]

        #     r̂e = eigsData[(j-1)*2+1]
        #     îm = eigsData[j*2]

        #     loss += error(func(re, im), func(r̂e, îm))
        # end
    end

    return loss
end

function gradStability(eigsArray; error=mae)
    loss = 0.0
    for i in 1:length(eigsArray) 
        eigs = eigsArray[i]
        
        numEigs = Int(length(eigs)/2)

        for j in 1:numEigs
            re = eigs[(j-1)*2+1]

            loss += error(re)
        end
    end
    return loss
end

function gradStiffness(eigsArray, stiffnessData; error=mae, eps=1e-8)
    loss = 0.0
    for i in 1:length(eigsArray) 
        eigs = eigsArray[i]
        
        loss += error(getStiffness(eigs), stiffnessData[i])
    end
    return loss
end

function gradOscillation(eigsArray, eigsArrayData; error=mae)
    loss = 0.0
    for i in 1:length(eigsArray) 

        eigs = eigsArray[i]
        eigsData = eigsArrayData[i]

        numEigs = Int(length(eigs)/2)

        #pairs = prepareEigPairs(eigs)
        matchedEigvals = zeros(Bool, numEigs)

        pairsData = prepareEigPairs(eigsData)
        numPairsData = length(pairsData)

        pairs = prepareEigPairs(eigs)
        numPairs = length(pairs)

        if numPairsData > numPairs
            for pd in 1:numPairsData
                for j in 1:numEigs

                    if matchedEigvals[j]
                        continue 
                    end

                    matchedEigvals[j] = true # j (will be) matched

                    minErrorRe = Inf
                    minErrorIndex = nothing 

                    re1 = eigs[(j-1)*2+1]
                    im1 = eigs[j*2]
                    
                    for k in 1:numEigs

                        if matchedEigvals[k] 
                            continue
                        end

                        re2 = eigs[(k-1)*2+1]
                        im2 = eigs[k*2]
                        
                        errorRe = error(re1, re2) 
                        
                        if errorRe < minErrorRe     # smaller? great!
                            minErrorRe = errorRe
                            minErrorIndex = k 
                        end

                        #loss += error
                    end

                    if minErrorIndex != nothing
                        loss += minErrorRe
                        matchedEigvals[minErrorIndex] = true
                    end

                end
            end
        elseif numPairs > numPairsData

            # while numPairs > numPairsData # remove easiest
            #     for j in 1:numPairs

            #         if matchedEigvals[j]
            #             continue 
            #         end

            #         re = eigs[(j-1)*2+1]
            #         im = eigs[j*2]

            #         loss += error(im, 0.0) 
            #     end
            # end

        else # balanced pairs! nothing to do here

        end

        # for the remaining (not merged) eigvals, punish the imaginary part:
        # - for non-paired eigenvalues, this does nothing (imaginary part is zero)
        # - for "too many" paired pairs, this forces separation (imaginary part is forced to zero)

        # for j in 1:numEigs
        #     re = eigs[(j-1)*2+1]
        #     im = eigs[j*2]

        #     r̂e = eigsData[(j-1)*2+1]
        #     îm = eigsData[j*2]

        #     loss += error(func(re, im), func(r̂e, îm))
        # end
    end

    return loss
end

function gradFrequency(args...; kwargs...)
    return gradEigCompare(getFrequency, args...; kwargs...)
end

function gradDamping(args...; kwargs...)
    return gradEigCompare(getDamping, args...; kwargs...)
end

function denan!(ar; substitute=0.0)
    for i in 1:length(ar)
        if isnan(ar[i])
            ar[i] = substitute
        end
    end
    return ar
end

function denan_count!(ar; substitute=0.0)
    numNaNs = 0
    for i in 1:length(ar)
        if isnan(ar[i])
            ar[i] = substitute
            numNaNs += 1
        end
    end
    return ar, numNaNs
end

const gradNames = ["SOL", "FRQ", "DMP", "STB", "OSC", "STF"]

struct RUN

    startTime::Float64

    iters::Vector{Int64}
    losses::Vector{Float64}
    lossesTest::Vector{Float64}
    times::Vector{Float64}
    params::Vector{ComponentVector{Float64}}
    
    function RUN()
        
        iters = Vector{Int64}()
        losses = Vector{Float64}()
        lossesTest = Vector{Float64}()
        times = Vector{Float64}()
        params = Vector{ComponentVector{Float64}}()

        return new(time(), iters, losses, lossesTest, times, params)
    end
end

struct EXPERIMENT

    id::Int32
    runs::Vector{RUN}
    
    function EXPERIMENT(_id::Integer)
        runs = Vector{RUN}()
        return new(_id, runs)
    end
end

function Base.push!(exp::EXPERIMENT)
    run = RUN()
    push!(exp.runs, run)
    return run 
end

function Base.push!(run::RUN, iter::Int64, loss::Float64, lossTest::Float64, params)

    t = time()

    push!(run.iters, iter)
    push!(run.losses, loss)
    push!(run.lossesTest, lossTest)
    push!(run.times, t - run.startTime)
    push!(run.params, deepcopy(params))
    
    nothing
end

function callback(state, val, num_iter::Integer, experiment::EXPERIMENT, loss, loss_parts, loss_test, gradScale, gradFilter, eta_start::Real, eta_stop::Real, eta_lambda::Real)
    
    p = state.u 
    iter = state.iter

    l = loss(p)

    if iter % 10 == 0
        run = experiment.runs[end]

        Optimisers.adjust!(state.original; eta = eta_stop + (eta_start-eta_stop) * exp(-eta_lambda*iter) )

        parts = loss_parts(p)
        l_sol = parts[1]
        l_test = loss_test(p)

        push!(run, iter, l_sol, l_test, p)

        str = "$(round(l; digits=6)) | $(iter)/$(num_iter) | $(round(iter/num_iter*100; digits=1))%"
        for i in 1:length(parts)
            str *= "\n$(gradNames[i])[$(gradFilter[i])]: $(round(parts[i]*gradScale[i]*gradFilter[i]; digits=6)) ($(round(parts[i]; digits=6)))"
        end
        @info str
    end

    return false
end

COLORS = Colors.distinguishable_colors(numStates+2, [RGB(1,1,1), RGB(0,0,0)])[3:end]
function plotImaginary!(fig, eigsArray, eigsData; limit::Real=0.0, kwargs...)
    limitX = limit 
    limitY = limit

    numEigs = Int(length(eigsArray[1])/2)

    for eigs in eigsArray[[1,end]]

        if limit == 0.0

            reals = collect(eigs[(i-1)*2+1] for i in 1:numEigs)
            imags = collect(eigs[i*2] for i in 1:numEigs)
            for i in 1:numEigs
                re = abs(reals[i])
                im = abs(imags[i])
                if re > limitX
                    limitX = re
                end
                if im > limitY 
                    limitY = im
                end
            end

        end
    end

    for eigs in eigsData[[1,end]]

        if limit == 0.0

            reals = collect(eigs[(i-1)*2+1] for i in 1:numEigs)
            imags = collect(eigs[i*2] for i in 1:numEigs)
            for i in 1:numEigs
                re = abs(reals[i])
                im = abs(imags[i])
                if re > limitX
                    limitX = re
                end
                if im > limitY 
                    limitY = im
                end
            end

        end
    end

    Plots.plot!(fig, [0, 0], [-limitY,limitY]; color=:black, style=:dash, label=:none, xlabel="Re", ylabel="Im", kwargs...)
    Plots.plot!(fig, [-limitX, limitX], [0,0]; color=:black, style=:dash, label=:none)

    # for j in 1:2
    #     eigs = eigsArray[[1,end]][j]

    #     reals = collect(eigs[(i-1)*2+1] for i in 1:numEigs)
    #     imags = collect(eigs[i*2] for i in 1:numEigs)

    #     realsData = collect(eigsData[j][(i-1)*2+1] for i in 1:numEigs)
    #     imagsData = collect(eigsData[j][i*2] for i in 1:numEigs)

    #     Plots.scatter!(fig, reals, imags, marker=:x, label="t=$(round((j-1)*10.0; digits=1))")
    #     Plots.scatter!(fig, realsData, imagsData, marker=:x, label="t=$(round((j-1)*10.0; digits=1)) | Data")
    # end

    for j in 1:numEigs
        # SIM
        reals = collect(eigs[(j-1)*2+1] for eigs in eigsArray)
        imags = collect(eigs[j*2] for eigs in eigsArray)

        Plots.plot!(fig, reals, imags, label="Eigval #$(j)", color=COLORS[j])

        # DATA
        reals = collect(eigs[(j-1)*2+1] for eigs in eigsData)
        imags = collect(eigs[j*2] for eigs in eigsData)

        Plots.plot!(fig, reals, imags, label="Eigval #$(j) [Data]", color=COLORS[j], style=:dash)
        Plots.scatter!(fig, [reals[end]], [imags[end]], marker=:x, label="Eigval #$(j) [Data, t=10.0s]", color=COLORS[j])
    end
end

function getGradFilterNames(gradFilters)
    gradFilterNames = []
    for gradFilter in gradFilters 
        name = ""

        for i in 1:6
            if gradFilter[i] == 1 
                if length(name) == 0
                    name *= gradNames[i]
                else
                    name *= "+" * gradNames[i]
                end
            end
        end

        push!(gradFilterNames, name)
    end
    return gradFilterNames
end

function startExperiment(foldername, maxRuns)

    expPath = joinpath(@__DIR__, foldername)
    
    ### MAKE FILE ###
    finished = false
    files = readdir(expPath)
    free_id = 1
    isFree = false
    # find first free ID
    while !isFree
        isFree = true
        for file in files
            name, ext = splitext(file)
            id = parse(Int32, name)
            if id == free_id
                free_id += 1
                isFree = false
                break
            end
        end
    end
    
    if free_id > maxRuns
        @info "Reached max runs $(maxRuns)"
        return nothing
    end

    JLD2.save(joinpath(@__DIR__, expPath, "$(free_id).jld2"), "finished", finished) 

    return EXPERIMENT(free_id)
end

function finishExperiment(foldername, exp::EXPERIMENT, args...)
    expPath = joinpath(@__DIR__, foldername)
    finished = true
    JLD2.save(joinpath(@__DIR__, expPath, "$(experiment.id).jld2"), "experiment", experiment, "finished", finished)
end

function loadExperiments(foldername)
    expPath = joinpath(@__DIR__, foldername)
    files = readdir(expPath)

    experiments = EXPERIMENT[]

    for file in files

        name, ext = splitext(file)
        if !endswith(lowercase(ext), "jld2")
            @warn "File $(file) is no JLD2!"
            continue 
        end

        fields = JLD2.load(joinpath(expPath, file))

        finished = fields["finished"]
        delete!(fields, "finished")

        if !finished
            @warn "File $(file) is not a finished run!"
            continue 
        end

        push!(experiments, fields["experiment"])

    end

    return experiments
end

# kill the script if it isn't the last experiment
function checkLastExperiment(foldername)
    expPath = joinpath(@__DIR__, foldername)
   
    files = readdir(expPath)
    for file in files

        name, ext = splitext(file)
        if !endswith(lowercase(ext), "jld2")
            @warn "File $(file) is no JLD2!"
            continue 
        end

        finished = JLD2.load(joinpath(expPath, file), "finished")

        @assert finished "Job for file `$(file)` still running, closing this script ..."
    end
end

function denan(a)
    b = copy(a)
    for i in 1:length(b)
        if isnan(b[i])
            if i > 1 
                b[i] = b[i-1]
            else
                j=1
                while isnan(b[i])
                    b[i] = b[i+j]
                    j+=1
                end
            end
        end
        @assert !isnan(b[i]) "b[i] still nan!"
    end
    return b
end

function getMedianTime(experiments::Vector{EXPERIMENT}, expIndex::Integer)
    dt_buf = []
    for experiment in experiments 
        run = experiment.runs[expIndex]
        dt = run.times[2:end] - run.times[1:end-1]
        push!(dt_buf, dt...)
    end

    return median(dt_buf)
end

function getLosses(experiments::Vector{EXPERIMENT}, expIndex::Integer)
    return collect(denan(experiment.runs[expIndex].losses[1:end-1]) for experiment in experiments)
end

function getLossesTest(experiments::Vector{EXPERIMENT}, expIndex::Integer)
    return collect(denan(experiment.runs[expIndex].lossesTest[1:end-1]) for experiment in experiments)
end

function getSolutions(neuralODE, experiments::Vector{EXPERIMENT}, expIndex::Integer, u0, st; step::Integer=length(experiments[expIndex].runs[runIndex].params))
    solutions = []
    for runIndex in 1:numRepetitions
        p = getParam(experiments, expIndex, runIndex; step=step)
        solution = neuralODE(u0, p, st)[1] 
        push!(solutions, solution)
    end
    return solutions
end

function getSolution(neuralODE, experiments::Vector{EXPERIMENT}, runIndex::Integer, expIndex::Integer, u0, st; step::Integer=length(experiments[expIndex].runs[runIndex].params))
    p = getParam(experiments, runIndex, expIndex; step=step)
    return neuralODE(u0, p, st)[1]
end

function getParams(experiments::Vector{EXPERIMENT}, expIndex::Integer; step::Integer=length(experiments[expIndex].runs[runIndex].params))
    ps = []
    for runIndex in 1:numRepetitions
        p = getParam(experiments, expIndex, runIndex; step=step)
        push!(ps, p)
    end
    return ps
end

function getParam(experiments::Vector{EXPERIMENT}, runIndex::Integer, expIndex::Integer; step::Integer=length(experiments[expIndex].runs[runIndex].params))
    return experiments[runIndex].runs[expIndex].params[step]
end

function getBestSolution(neuralODE, experiments::Vector{EXPERIMENT}, expIndex::Integer, u0, st)
    losses = getLosses(experiments, expIndex)
    bestIndex = 1
    for i in 2:length(losses)
        if losses[i][end] < losses[bestIndex][end]
            bestIndex = i
        end
    end
    p = experiments[bestIndex].runs[expIndex].params[end]
    return neuralODE(u0, p, st)[1]
end

function getMedianSolution(neuralODE, experiments::Vector{EXPERIMENT}, expIndex::Integer, u0, st)
    losses = getLossesTest(experiments, expIndex)

    med_end = median(collect(losses[j][end] for j in 1:length(losses)))
    
    bestIndex = 1
    for i in 2:length(losses)
        if abs(losses[i][end] - med_end) < abs(losses[bestIndex][end] - med_end)
            bestIndex = i
        end
    end

    p = experiments[bestIndex].runs[expIndex].params[end-1]

    return neuralODE(u0, p, st)[1]
end

using LinearAlgebra
function getMaxRes(neuralODE, experiments, expIndex::Integer, u0, st)

    ret = []

    numRepetitions = length(experiments)
    iters = experiments[1].runs[1].iters

    for runIndex in 1:numRepetitions

        maxRes = []

        for j in 1:length(iters)

            solution = getSolution(neuralODE, experiments, runIndex, expIndex, u0, st; step=j)
            p = getParam(experiments, runIndex, expIndex; step=j)

            maxRe = -Inf 
            f = u -> neuralODE.model(u, p, st)[1]

            _eigvalData = collect(eigvals(ForwardDiff.jacobian(f, u)) for u in solution.u)

            for eigvals in _eigvalData
                for eigval in eigvals 
                    if real(eigval) > maxRe 
                        maxRe = real(eigval)
                    end
                end
            end
            push!(maxRes, maxRe)
        end

        push!(ret, maxRes)
    end

    return ret
end

# array [REPETITIONS, STPES]
function stats(array)
    ne = length(array)
    steps = 1:length(array[1])
    med = collect(    median(collect(array[j][s] for j in 1:ne)    ) for s in steps)
    p25 = collect(percentile(collect(array[j][s] for j in 1:ne), 25) for s in steps) 
    p75 = collect(percentile(collect(array[j][s] for j in 1:ne), 75) for s in steps) 
    return p25, med, p75
end

function plotContributions(neuralODE, experiments, expIndex::Integer, u0, st; skipFirst=0)
    
    iters = experiments[1].runs[expIndex].iters
    steps = 1:length(iters)
    cols = 6

    colors = Colors.distinguishable_colors(cols, [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)

    gradFilter = gradFilters[expIndex]
    fig = plot(; title="#$(expIndex) | $(gradFilterNames[expIndex])")
    for i in 1:cols

        if gradFilter[i] == 0
            continue 
        end

        vals = []
        for runIndex in 1:numRepetitions
            buf = []
            for j in steps
                p = getParam(experiments, runIndex, expIndex; step=j)
                push!(buf, loss_parts(neuralODE, u0, p, st, gradFilter)[i] * gradScale[i])
            end
            push!(vals, buf)
        end

        # vals = RUNS -> STEPS
        p25, med, p75 = stats(vals)
        mea = mean(med)
        r25 = med - p25 
        r75 = p75 - med
        
        # for j in 1:length(med)
        #     if med[j] <= 0.0# || (med[j] - r25[j]) <= 0.0
        #         med[j] = Inf # med[j-1]
        #         r25[j] = 0.0 # p25[j-1]
        #         r75[j] = 0.0 # p75[j-1]
        #     end
        # end

        # global a, b, c 
        # a = r25 
        # b = med
        # c = r75

        #@assert min(med...) > 0.0 "Log error < 0"
        #@assert min((med-r25)...) > 0.0 "Log r25 error < 0"
        #@assert min((med+r75)...) > 0.0 "Log r75 error < 0"

        plot!(fig, iters, med; ribbon=(r25, r75), fillalpha=0.2, label="$(gradNames[i]) $(mea)", color=colors[i])
        #display(fig)
    end
    return fig
end

function setup(tspan=tspan; saveat=saveat, kwargs...)
    net = Chain(preprocess,
        Dense(numStates, layerwidth, tanh),
        Dense(layerwidth, Int(numStates/2)),
        postprocess)
    p, st = Lux.setup(rng, net)
    solver = Tsit5()
    neuralODE = NeuralODE(net, tspan, solver; saveat=saveat, kwargs...)

    return net, p, st, solver, neuralODE
end

function plotConvergence(filename, experiments)
    gradCOLORS = Colors.distinguishable_colors(length(gradFilters), [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)
    linewidth = 2
    ne = length(experiments)

    best_sol = 0.0
    iters = experiments[1].runs[1].iters[1:end-1]

    fig = startPlot(;ylabel=L"l_{SOL}~[\mathrm{m}]", xlabel=L"t~[\mathrm{s}]", legend=:topright, yaxis=:log);
    for gf in 1:length(gradFilters)

        losses = getLosses(experiments, gf)
        steps = 1:length(losses[1])

        med = collect(    median(collect(losses[j][s] for j in 1:ne)    ) for s in steps)
        p25 = collect(percentile(collect(losses[j][s] for j in 1:ne), 25) for s in steps) 
        p75 = collect(percentile(collect(losses[j][s] for j in 1:ne), 75) for s in steps) 

        if gf == 1
            best_sol = med[end]
        end

        Plots.plot!(fig, iters, med; ribbon=((med-p25), (p75-med)), fillalpha=0.2, color=gradCOLORS[gf], linewidth=linewidth, label=gradFilterNames[gf])
    end

    Plots.plot!(fig, [iters[1], iters[end]], [best_sol, best_sol]; color=gradCOLORS[1], style=:dash, linewidth=linewidth, label="SOL (final value)")

    savefig(fig, joinpath(@__DIR__, "imgs", filename))
    fig
end

function plotMedian(filename, experiments, u0; tspan=(tspan[1], tspan[end]*2))
    gradCOLORS = Colors.distinguishable_colors(length(gradFilters), [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)
    linewidth = 2
    ne = length(experiments)

    saveat = (tspan[1]:0.01:tspan[end])

    fig = startPlot(;
        ylabel=L"s~[\mathrm{m}]", 
        xlabel=L"t~[\mathrm{s}]", 
        legend=:bottomright, 
        size=(540/0.75,360),
        bottom_margin = 2Plots.mm,
        left_margin = 2Plots.mm
        );

    for gf in 1:length(gradFilters)

        net, _, st, solver, neuralODE = setup(tspan; saveat=saveat)

        #for i in 1:length(experiments)
        #    solution = getSolution(neuralODE, experiments, gf, i, u0, st)
        #    Plots.plot!(fig, solution.t, collect(u[1] for u in solution.u); label=(i == 1 ? "$(gradFilterNames[gf])" : :none), color=gradCOLORS[gf], linewidth=linewidth, linealpha=0.25)
        #end

        solution = getMedianSolution(neuralODE, experiments, gf, u0, st)

        Plots.plot!(fig, solution.t, collect(u[1] for u in solution.u); label="$(gradFilterNames[gf])", color=gradCOLORS[gf], linewidth=linewidth)
    end

    coarse_saveat = range(saveat[1], saveat[end], length(posData)*2-1)

    refProb = ODEProblem{false}(refODE, u0, tspan);
    refSol = solve(refProb; saveat=saveat, u0=u0)
    Plots.plot!(fig, refSol.t, collect(u[1] for u in refSol.u); label="ground truth", color=:black, style=:dash, linewidth=linewidth)
    Plots.scatter!(fig, coarse_saveat, collect(refSol(t)[1] for t in coarse_saveat); label="ground truth data", color=:black, linewidth=linewidth)
    Plots.plot!(fig, [tspan[end]/2, tspan[end]/2], [-u0[1]*1.1, u0[1]*1.1]; label=:none, color=:black)
    savefig(fig, joinpath(@__DIR__, "imgs", filename))

    fig
end

function plotStability(filename, experiments)
    gradCOLORS = Colors.distinguishable_colors(length(gradFilters), [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)
    linewidth = 2
    ne = length(experiments)

    fig = startPlot(;ylabel=L"\Re(\lambda_{w})~\left[\frac{1}{\mathrm{s}}\right]", xlabel=L"\mathrm{steps}", legend=:topright);
    for gf in 1:length(gradFilters)

        net, _, st, solver, neuralODE = setup()

        maxRes = getMaxRes(neuralODE, experiments, gf, u0, st)

        steps = 1:length(maxRes[1])-1
        iters = experiments[1].runs[1].iters[1:end-1]

        med = collect(    median(collect( maxRes[j][s] for j in 1:ne)    ) for s in steps)
        p25 = collect(percentile(collect( maxRes[j][s] for j in 1:ne), 25) for s in steps) 
        p75 = collect(percentile(collect( maxRes[j][s] for j in 1:ne), 75) for s in steps) 

        Plots.plot!(fig, iters, med; ribbon=(med-p25, p75-med), fillalpha=0.2, label=gradFilterNames[gf], color=gradCOLORS[gf], linewidth=linewidth)
    end
    Plots.plot!(fig, [0,numSteps], [0, 0]; label="border stable", color=:black, style=:dash, linewidth=linewidth)
    savefig(fig, joinpath(@__DIR__, "imgs", filename))
    fig
end

function latexResults(filename, experiments)
    ne = length(experiments)
    open(joinpath(@__DIR__, "latex", filename * "_specs.tex"), "w") do io
        println(io, "\\begin{tabular}{lllll}")
        println(io, "\\toprule")
        println(io, "Loss & Sim. & Grad. & Acc. & Rej. \\\\")
        println(io, "~ & Time [ms] & Time [s] & Steps & Steps \\\\")
        println(io, "\\midrule")

        iters = experiments[1].runs[1].iters
        steps = 1:length(iters)

        for gf in 1:length(gradFilters)
            
            net, _, st, solver, neuralODE = setup()

            # training online
            
            med_sol = nothing
            dts = Vector{Float64}(undef, 1000)
            for i in 1:1000
                ts = time_ns()/1e6
                med_sol = getMedianSolution(neuralODE, experiments, gf, u0, st)
                dts[i] = (time_ns()/1e6-ts)
            end

            med_time = getMedianTime(experiments, gf)
            med_sim = median(dts)
           
            println(io, "$(gradFilterNames[gf]) & $(round(med_sim; digits=2)) & $(round(med_time; digits=2)) & $(med_sol.destats.naccept) & $(med_sol.destats.nreject) \\\\")
        end

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end

    open(joinpath(@__DIR__, "latex", filename * "_train.tex"), "w") do io
        println(io, "\\begin{tabular}{lllll}")
        println(io, "\\toprule")
        println(io, "Loss & \$P_{25}\$ & Median & \$P_{75}\$ & Hit \\\\")
        println(io, "\\midrule")

        best_sol = 0.0
        iters = experiments[1].runs[1].iters

        for gf in 1:length(gradFilters)
            losses = getLosses(experiments, gf)
            steps = 1:length(losses[1])
            
            net, _, st, solver, neuralODE = setup()

            med = collect(    median(collect(losses[j][s] for j in 1:ne)    ) for s in steps)
            p25 = collect(percentile(collect(losses[j][s] for j in 1:ne), 25) for s in steps) 
            p75 = collect(percentile(collect(losses[j][s] for j in 1:ne), 75) for s in steps)
            
            hit_step = 0
            if gf == 1 
                best_sol = med[end]
            else
                for i in steps
                    if med[i] < best_sol
                        hit_step = iters[i]
                        break 
                    end
                end
            end

            println(io, "$(gradFilterNames[gf]) & $(round(p25[end]; digits=4)) & $(round(med[end]; digits=4)) & $(round(p75[end]; digits=4)) & $((hit_step == 0 ? "n.a." : hit_step))\\\\")
        end

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end

    open(joinpath(@__DIR__, "latex", filename * "_test.tex"), "w") do io
        println(io, "\\begin{tabular}{llll}")
        println(io, "\\toprule")
        println(io, "Loss & \$P_{25}\$ & Median & \$P_{75}\$ \\\\")
        println(io, "\\midrule")

        for gf in 1:length(gradFilters)
            losses = getLossesTest(experiments, gf)
            steps = 1:length(losses[1])
            iters = experiments[1].runs[1].iters

            net, _, st, solver, neuralODE = setup()

            med = collect(    median(collect(losses[j][s] for j in 1:ne)    ) for s in steps)
            p25 = collect(percentile(collect(losses[j][s] for j in 1:ne), 25) for s in steps) 
            p75 = collect(percentile(collect(losses[j][s] for j in 1:ne), 75) for s in steps)
            
            println(io, "$(gradFilterNames[gf]) & $(round(p25[end]; digits=4)) & $(round(med[end]; digits=4)) & $(round(p75[end]; digits=4)) \\\\")
        end

        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end