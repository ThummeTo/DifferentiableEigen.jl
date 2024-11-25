#
# Copyright (c) 2023 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# 🚧 Disclaimer: This is draft code 🚧 #
# please check Project.toml for compatibility #

using Flux
import SciMLBase: RightRootFind
using DifferentialEquations
using DiffEqCallbacks
using DiffEqFlux: ODEFunction, basic_tgrad, ODEProblem, ZygoteVJP, InterpolatingAdjoint, solve, NeuralODE
using Plots
import ForwardDiff
import LinearAlgebra: eigvals, eigvecs, det, UpperTriangular, I, Diagonal, rank, norm
import LinearAlgebra
import FMIBase: undual, isdual
using Flux.Losses:mse
import Colors
using Printf: @sprintf
using LaTeXStrings
using NLsolve
using Colors

function prepareEigPairs(eigvalData::AbstractVector{<:AbstractVector}; eps=1e-16)
    eigsPairs = Vector{Vector{Int}}[]
    for eigs in eigvalData
        pairs = prepareEigPairs(eigs)
       
        push!(eigsPairs, pairs)
        
    end
    return eigsPairs
end

function prepareEigPairs(eigs::AbstractVector{<:Real}; eps=1e-16)
    
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
function f(x, p, t)
    return neuralODE.re(p)(x)
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

using ChainRulesCore

function _eigen(A::AbstractMatrix)
    A = undual(A)
    val, vec = LinearAlgebra.eigen(A; sortby=LinearAlgebra.eigsortby)
    
    @assert !isdual(val) "!!!"
    @assert !isdual(vec) "!!!"

    return comp2Arr(val), comp2Arr(vec) # return real.(val), real.(vec)
end

function ChainRulesCore.frule((Δself, ΔA), ::typeof(_eigen), A::AbstractMatrix)
    #@info "frule start"

    A = undual(A)
    
    eV = LinearAlgebra.eigen(A; sortby=LinearAlgebra.eigsortby)
    e,V = eV
    n = size(A,1)

    Ω = comp2Arr(e), comp2Arr(V) # Ω = real.(e), real.(V)
    ∂e = ZeroTangent()
    ∂V = ZeroTangent()

    #D = Diagonal(e)
    U = V
    F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]

    @assert size(U) == size(ΔA)

    UAU = ΔA

    if rank(U) == n
        UAU = inv(U) * ΔA * U
    end

    ∂e = LinearAlgebra.diag(UAU) # exact: I .* UAU
    ∂V = U * (F .* UAU)

    ∂Ω = (comp2Arr(∂e), comp2Arr(∂V))  # (real.(∂e), real.(∂V)) 

    #@info "frule end |∂e| = $(sum(∂e)), |∂V| = $(sum(∂V))"

    @assert !isdual(∂e) "!!!"
    @assert !isdual(∂V) "!!!"
    @assert !isdual(Ω[1]) "!!!"
    @assert !isdual(Ω[2]) "!!!"

    return Ω, ∂Ω 
end

function ChainRulesCore.rrule(::typeof(_eigen), A::AbstractMatrix)
    
    eU = eigen(A)
    e,U = eU
    n = size(A,1)

    Ω = eU

    function pullback(r̄)

        ē, Ū = r̄

        Ā = ZeroTangent()

        D̄ = nothing 
        
        if ē != nothing
            D̄ = Diagonal(ē)
        end
        
        if Ū === nothing
            Ā = inv(U)' * D̄ * U'

        elseif D === nothing
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
@ForwardDiff_frule _eigen(A::AbstractMatrix{<:ForwardDiff.Dual})

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

global eigsRe = zeros(numStates)
global eigsIm = zeros(numStates)

function trackEigvals(eigs)
    global eigsRe, eigsIm

    numEigs = Int(length(eigs)/2)

    tracked = zeros(Bool, numEigs)
    order = []
    
    for i in 1:numEigs

        re_i = eigsRe[i] 
        im_i = eigsIm[i]

        closest = 0 
        closestDist = Inf

        for j in 1:numEigs
            if !tracked[j]

                re_j = eigs[(j-1)*2+1]
                im_j = eigs[j*2]
                dist = sqrt((re_i-re_j)^2 + (im_i-im_j)^2)

                if dist < closestDist
                    closest = j
                    closestDist = dist
                end

            end
        end

        tracked[closest] = true
        push!(order, closest)

        eigsRe[i] = eigs[(closest-1)*2+1]
        eigsIm[i] = eigs[closest*2]
    end

    return order
end

global eigsArray = nothing
# MSE between net output and data
function losssum(p)
    global solution, lossReal, lossImag, lossStab, lossPos, loss, eigsArray

    solution = solve(prob, neuralODE.args...; saveat=saveat, neuralODE.kwargs..., p=p) # sensealg=sensealg, 

    t = 0.0
    _f = _x -> f(_x, p, t)

    eigsArray = collect(_eigen(ForwardDiff.jacobian(_f, x))[1] for x in solution.u)

    lossSolution = gradSolution(solution; error=(x, y)->abs(x-y).*gradScale[1] / numTs )

    lossFrequency = gradFrequency(eigsArray, eigvalData; error=(x, y)->abs(x-y)*gradScale[2] / numEigs /numTs )
    lossDamping = gradDamping(eigsArray, eigvalData; error=(x, y)->abs(x-y)*gradScale[3] / numEigs /numTs)
    lossStability = gradStability(eigsArray; error= (x) -> max(x, 0.0)*gradScale[4] / numEigs / numTs)
    lossOscillation = gradOscillation(eigsArray, eigvalData; error=(x, y)->abs(x-y)*gradScale[5] / numEigs /numTs) # , eigsPairs
    lossStiffness = gradStiffness(eigsArray, stiffnessData; error=(x, y)->abs(x-y)*gradScale[6] /numTs )
    
    return [lossSolution, lossFrequency, lossDamping, lossStability, lossOscillation, lossStiffness]
end

function gradSolution(solution; error=mse)
    posNet = collect(data[1] for data in solution.u) 
    velNet = collect(data[2] for data in solution.u)

    pos2Net = nothing 
    vel2Net = nothing
    if numStates == 4
        pos2Net = collect(data[3] for data in solution.u) 
        vel2Net = collect(data[4] for data in solution.u)
    end

    lossSolution = sum(collect(error(posNet[i], posData[i]) for i in 1:numTs)) 

    if numStates == 4 
        lossSolution += sum(collect(error(pos2Net[i], pos2Data[i]) for i in 1:numTs)) 
    end

    return lossSolution
end

function getFrequency(re, im)
    #return sqrt(re^2 + im^2)
    return abs(im) / (2.0 * π) 
end

function getDamping(re, im; eps=1e-16)
    #return cos(atan(im, re))
    len = sqrt(re^2 + im^2)
    if len < eps
        return 0.0
    else
        return -re / len
    end 
end

function getStiffness(eigs, eps=1e-32)
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

    if _min > eps 
        return _max/_min
    else 
        return Inf
    end
end

function gradEigCompare(func, eigsArray, eigsArrayData; error=mse)
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

function gradStability(eigsArray; error=mse)
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

function gradStiffness(eigsArray, stiffnessData; error=mse, eps=1e-32)
    loss = 0.0
    for i in 1:length(eigsArray) 
        eigs = eigsArray[i]
        
        loss += error(getStiffness(eigs), stiffnessData[i])
    end
    return loss
end

function gradOscillation(eigsArray, eigsArrayData; error=mse)
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

gradNames = ["SOL", "FRQ", "DMP", "STB", "OSC", "STF"]

function train!(p_net, optim; epoch=25, numSteps=10000, reinitat=-1, gradMode=:GradMulti, gradFilter=[   1,     1,     1,     0,     1,     0], gradScale=[  1e0,  1e-1,   1e0,   1e3,   1e1,  1e0])
    bef = losssum(p_net[1])
    bef[1]

    losses = []
    solutions = []
    params = []

    push!(losses, (0, bef))
    push!(solutions, solution)
    push!(params, copy(p_net[1]))

    anim = @animate for i in 1:round(Integer, numSteps/epoch)

        if i == reinitat
            @info "Reinit optimizer"
            optim = Adam(1e-2)
        end

        grads = nothing
        grad = nothing
        usedGrad = nothing
        numNaNs = zeros(Integer, 6)

        comp = (g) -> sum(abs.(g))
        #comp = (g) -> max(abs.(g)...)

        for e in 1:epoch

            for j in 1:length(p_net)
                
                jac = ForwardDiff.jacobian(losssum, p_net[j])
                
                gradSolution, numNaNs[1] = denan_count!(jac[1,:])
                gradFrequency, numNaNs[2] = denan_count!(jac[2,:])
                gradDamping, numNaNs[3] = denan_count!(jac[3,:])
                gradStability, numNaNs[4] = denan_count!(jac[4,:])
                gradOscillation, numNaNs[5] = denan_count!(jac[5,:])
                gradStiffness, numNaNs[6] = denan_count!(jac[6,:])

                usedGrad = zeros(Integer, 6)

                grads = [gradSolution, gradFrequency, gradDamping, gradStability, gradOscillation, gradStiffness] .* gradFilter
                grad = nothing

                # opt a
                if gradMode == :GradMix

                    grad = zeros(length(grads[1]))

                    for k in 1:length(grad)
                        for g in 1:length(grads)
                            if gradFilter[g] > 0
                                if abs(grads[g][k]) > abs(grad[k])
                                    grad[k] = grads[g][k]
                                    usedGrad[g] = 1
                                end
                            end
                        end
                    end

                elseif gradMode == :GradSwitch
                # opt b
                
                    grad = zeros(length(grads[1]))

                    gradInd = 0
                    for g in 1:length(grads)
                        if gradFilter[g] > 0
                            if comp(grads[g]) > comp(grad)
                                grad = copy(grads[g])
                                gradInd = g
                            end
                        end
                    end

                    usedGrad[gradInd] = 1

                elseif gradMode == :GradSum # opt C
                
                    grad = zeros(length(grads[1]))

                    for g in 1:length(grads)
                        if gradFilter[g] > 0
                            grad += grads[g]
                            usedGrad[g] = 1
                        end
                    end
                    
                elseif gradMode == :GradOrig # opt D
                
                    grad = copy(grads[1])
                    usedGrad[1] = 1

                elseif gradMode == :GradMulti 

                    for g in 1:length(grads)
                        if gradFilter[g] > 0 && comp(grads[g]) > 0.0
                            tmpGrad = copy(grads[g])
                            step = Flux.Optimise.apply!(optim, p_net[j], tmpGrad)
                            p_net[j] .-= step
                            usedGrad[g] = 1
                        end
                    end
                end

                if grad != nothing # gradMode != :GradMulti 
                    step = Flux.Optimise.apply!(optim, p_net[j], grad)
                end

                p_net[j] .-= step
            end
        end

        closs = losssum(p_net[1])

        push!(losses, (i*epoch, closs))
        push!(solutions, solution)
        push!(params, copy(p_net[1])) 

        str = "$(gradMode) [$(i*epoch) / $(numSteps)]\n------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
        for g in 1:length(grads) 
            str *= @sprintf("[%s][%d][%d] Min = %16.4f | Max = %16.4f | Val = %16.4f |    Loss = %16.4f |    Loss (Norm.) = %16.4f |   NaNs: %d\n", gradNames[g], gradFilter[g], usedGrad[g], min(abs.(grads[g])...), max(abs.(grads[g])...), comp(grads[g]), closs[g], closs[g] ./ gradScale[g], numNaNs[g])
        end

        str *= "------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
        if grad != nothing
            str *= @sprintf("[RES]       Min = %16.4f | Max = %16.4f | Val = %16.4f | SumLoss = %16.4f | SumLoss (Norm.) = %16.4f\n", min(abs.(grad)...), max(abs.(grad)...), comp(grad), sum(denan!(closs) .* gradFilter), sum(denan!(closs) .* gradFilter ./ gradScale))
        else
            str *= @sprintf("[RES]                                                                                | SumLoss = %16.4f | SumLoss (Norm.) = %16.4f\n\n", sum(denan!(closs) .* gradFilter), sum(denan!(closs) .* gradFilter ./ gradScale))
        end
        @info "$str"

        # Solution plotting
        fig = startPlot(; layout=(3,1), size=(480, 480*3))
        plot!(fig[1], saveat, collect(ForwardDiff.value(u[1]) for u in solution.u), label="s NeuralODE", title="Training step $(i*epoch) ($(gradMode), reinit=$(reinitat))"); # , ylims=(-x0[1],x0[1])
        plot!(fig[1], saveat, posData, label="s Data")

        if numStates == 4
            plot!(fig[1], saveat, collect(ForwardDiff.value(u[3]) for u in solution.u), label="s2 NeuralODE"); # , ylims=(-x0[1],x0[1])
            plot!(fig[1], saveat, pos2Data, label="s2 Data")
        end

        # Eigs plotting
        plotImaginary!(fig[2], eigsArray, eigvalData)

        # Loss plotting
        s = zeros(length(losses))
        plot!(fig[3]; legend=:bottomleft) # yaxis=:log, 
        for g in 1:length(grads) 
            if gradFilter[g] > 0
                vals = collect(e[2][g] for e in losses)
                divider = max(vals...)
                plot!(fig[3], collect(e[1] for e in losses), vals ./ divider, label="$(gradNames[g])")
                s += vals
            end
        end

        divider = max(s...)
        plot!(fig[3], collect(e[1] for e in losses), s ./ divider, style=:dash, label="SUM")
        
        display(fig)
    end

    return anim, losses, solutions, params
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