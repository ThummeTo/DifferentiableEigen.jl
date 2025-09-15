#
# Copyright (c) 2023 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# 🚧 Disclaimer: This is draft code 🚧 #
# please check Project.toml for compatibility #

numStates = 2
include(joinpath(@__DIR__, "main.jl"))
using LaTeXStrings, Colors

# params
foldername = "results_cost"
posScale = 1.0
velScale = 1.0

# consts 
tspan = (0.0, 10.0)
saveat = 0.0:0.25:10.0

posData = zeros(length(saveat))
velData = zeros(length(saveat))
pos2Data = zeros(length(saveat))
vel2Data = zeros(length(saveat))

# sys
# c = (1 * 2*π)^2 / 1
# p = [c, 0.05]

# function create_sys(n, p)
#     sys = function(x)
#         dx = zeros(n*2)
#         for i in 1:n 
#             dx[1+(i-1)*2:i*2] = translational_pendulum(x[1+(i-1)*2:i*2], p)
#         end
#         return dx
#     end
#     return sys
# end

function create_u0(n)
    x0 = zeros(n*2)
    for i in 1:n 
        x0[1+(i-1)*2] = 1.0
    end
    return x0 
end

using ForwardDiff, BenchmarkTools

ns = 1:50 # 25
b_nodes = []
b_eig21_nodes = []
b_eig41_nodes = []

global gradFilter, _loss, p_orig

for n in ns

    @info "$(n) / $(ns[end])"

    global gradFilter, _loss, p_orig
    global eigvalData, saveat

    #sys = create_sys(n, p)
    u0 = create_u0(n)
    #sys(u0)

    # setup ANN
    net = Chain(Dense(n*2, layerwidth, tanh),
        Dense(layerwidth, n*2))
    p_orig, st = Lux.setup(rng, net)
    p_orig = Float64.(ComponentArray(p_orig))

    gradFilter_eig_node = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    gradFilter_node = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    solver = Tsit5()

    # pure NODE
    saveat = 0.0:0.25:10.0
    eigvalData = collect(zeros(n*2) for t in saveat)
    neuralODE = NeuralODE(net, tspan, solver; saveat=saveat)
    gradFilter = gradFilter_node
    _loss_parts = p -> loss_parts(neuralODE, u0, p, st, gradFilter)
    _loss = p -> loss(neuralODE, u0, p, st, loss_parts, gradFilter, gradScale)
    
    b_node = @benchmark ForwardDiff.gradient(_loss, p_orig)
    push!(b_nodes, median(b_node).time / 1e9)

    # eig NODE 41
    # saveat = 0.0:0.25:10.0
    # neuralODE = NeuralODE(net, tspan, solver; saveat=saveat)
    gradFilter = gradFilter_eig_node
    _loss_parts = p -> loss_parts(neuralODE, u0, p, st, gradFilter)
    _loss = p -> loss(neuralODE, u0, p, st, loss_parts, gradFilter, gradScale)

    b_eig41_node = @benchmark ForwardDiff.gradient(_loss, p_orig)
    push!(b_eig41_nodes, median(b_eig41_node).time / 1e9)

    # eig NODE 21
    saveat = 0.0:0.5:10.0
    eigvalData = collect(zeros(n*2) for t in saveat)
    neuralODE = NeuralODE(net, tspan, solver; saveat=saveat)
    #gradFilter = gradFilter_eig_node
    _loss_parts = p -> loss_parts(neuralODE, u0, p, st, gradFilter)
    _loss = p -> loss(neuralODE, u0, p, st, loss_parts, gradFilter, gradScale)

    b_eig21_node = @benchmark ForwardDiff.gradient(_loss, p_orig)
    push!(b_eig21_nodes, median(b_eig21_node).time / 1e9)
end

function fun(x, p)
    a, b = p 

    #a = abs(a)
    #b = abs(b)

    y = a * (x^b)
    return y 
end

function loss(p, comp)
    sum = 0.0
    for i in 1:length(ns)
        x = ns[i]
        y = fun(x, p)
        ŷ = comp[i]
        sum += abs(y-ŷ) # ^2
    end
    return sum 
end

using Optimization, Zygote

function clip!(vec, val)
    for i in 1:length(vec)
        if abs(vec[i]) > val 
            vec[i] = sign(vec[i])*val 
        end
    end
    nothing
end

import OptimizationOptimisers
function optim(loss)
    p = [1.0, 1.0]
    optf = OptimizationFunction((u, p) -> loss(u), AutoZygote())
    prob = OptimizationProblem(optf, p)
    sol = solve(prob, OptimizationOptimisers.Adam(1e-2); maxiters=1e4)
    return sol.u
end

# fig = plot(ns .* 2, b_nodes; xlabel="number of states", ylabel="time [s]", label="w.o. eigen-informed", width=2, yaxis=:log, legend=:topleft)
# plot!(fig, ns .* 2, b_eig21_nodes; label="w. eigen-informed [21]", width=2)
# plot!(fig, ns .* 2, b_eig41_nodes; label="w. eigen-informed [41]", width=2)
# savefig(fig, joinpath(@__DIR__, "times_log.pdf"))

COLORS = distinguishable_colors(4, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
COLORS_ITER = 2

fig = plot(ns .* 2, b_nodes; xlabel="n (number of states)", ylabel="time [s]", label="w.o. eigen-informed", width=2, legend=:topleft, color=COLORS[COLORS_ITER], yaxis=:log)
_loss = _p -> loss(_p, b_nodes)
p = optim(_loss)
plot!(fig, ns .* 2, collect(fun(x,p) for x in ns); label="$(round(p[1]; digits=3)) * n ^ $(round(p[2]; digits=2))", width=2, style=:dash, color=COLORS[COLORS_ITER])
COLORS_ITER += 1

plot!(fig, ns .* 2, b_eig21_nodes; label="w. eigen-informed [21]", width=2, color=COLORS[COLORS_ITER])
_loss = _p -> loss(_p, b_eig21_nodes)
p = optim(_loss)
plot!(fig, ns .* 2, collect(fun(x,p) for x in ns); label="$(round(p[1]; digits=3)) * n ^ $(round(p[2]; digits=2))", width=2, style=:dash, color=COLORS[COLORS_ITER])
COLORS_ITER += 1

plot!(fig, ns .* 2, b_eig41_nodes; label="w. eigen-informed [41]", width=2, color=COLORS[COLORS_ITER])
_loss = _p -> loss(_p, b_eig41_nodes)
p = optim(_loss)
plot!(fig, ns .* 2, collect(fun(x,p) for x in ns); label="$(round(p[1]; digits=3)) * n ^ $(round(p[2]; digits=2))", width=2, style=:dash, color=COLORS[COLORS_ITER])

savefig(fig, joinpath(@__DIR__, "times.pdf"))