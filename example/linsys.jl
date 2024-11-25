#
# Copyright (c) 2023 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# 🚧 Disclaimer: This is draft code 🚧 #
# please check Project.toml for compatibility #

numStates = 2
include(joinpath(@__DIR__, "main.jl"))

# consts 
tspan = (0.0, 10.0)
saveat = 0.0:0.1:10.0

p = [25.0, 0.05]
sys = (x) -> translational_pendulum(x, p)
x0 = [1.0, 0.0] 
reff = (x, p, t) -> sys(x) 

refODE = ODEFunction{false}(reff,tgrad=basic_tgrad);
refProb = ODEProblem{false}(refODE, x0, tspan);
refSol = solve(refProb;saveat=saveat)
plot(refSol)

# data gathering
eigvalData = collect(_eigen(ForwardDiff.jacobian(sys, u))[1] for u in refSol.u)
isLinear = eigvalData[1] == eigvalData[end]
stiffnessData = prepareStiffness(eigvalData)
eigsPairs = prepareEigPairs(eigvalData)

posData = collect(u[1] for u in refSol.u)
velData = collect(u[2] for u in refSol.u)

numStates = length(x0)
numEigs = Int(length(eigvalData[1])/2)
numTs = length(saveat)

# setup an almost neutral neural network (so we still can see the original system, even if we have some ANN added)
st = 32
global v1, v2
function preprocess(dx)
    global v1, v2

    v1 = dx[2]
    if numStates == 4
        v2 = dx[4]
    end

    return dx 
end
function postprocess(a)
    global v1, v2

    if numStates == 2
        return [v1, a[1]]
    else
        return [v1, a[1], v2, a[2]]
    end
end

d1 = rand(st,numStates)
d2 = rand(Int(numStates/2), st)

# Experiment START

gradFilterSolutions = []
gradFilterLosses = []
gradFilterParams = []

#              ["SOL", "FRQ", "DMP", "STB", "OSC", "STF"]
gradFilters = [[   1,     0,     0,     0,     0,     0],
               [   1,     0,     0,     1,     0,     0],
               [   1,     0,     0,     0,     0,     1],
               [   1,     0,     0,     1,     1,     0],
               [   1,     1,     1,     0,     1,     0],
               [   1,     1,     1,     1,     1,     0],
               [   1,     1,     1,     1,     1,     1]]

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

epoch = 25 # 25
numSteps = 5000 # 5000 # 15000
reinitat = -1 # 100
gradMode = :GradMulti # :GradMix, :GradSwitch, :GradSum, :GradOrig, :GradMulti
gradScale = [  1e0,  1e1,   1e0,   1e3,   1e1,  1e-1] 

for gf in 1:length(gradFilters)

    global neuralODE, prob

    gradFilter = gradFilters[gf]

    net = Chain(preprocess,
                Dense(d1, zeros(st), tanh),
                Dense(d2, zeros(Int(numStates/2))),
                postprocess)

    solver = Tsit5()

    # ForwardDiffSensitivity, Interpolatiing Adjoints and QuadratureAdjoints (only ForwardDiffSensitivity is working for me)
    neuralODE = NeuralODE(net, tspan, solver; saveat=saveat)

    # OUT OF PLACE # 
    dudt_op(x, p, t) = f(x, p, t);
    ff = ODEFunction{false}(dudt_op,tgrad=basic_tgrad);
    prob = ODEProblem{false}(ff, x0, tspan, neuralODE.p);

    p_net = Flux.params(neuralODE)
    optim = Adam(1e-3) # Flux.Optimiser(Adam(1e-2), ExpDecay(1.0, 0.9, 100, 1e-1))

    anim, losses, solutions, params = train!(p_net, optim; epoch=epoch, numSteps=numSteps, reinitat=reinitat, gradMode=gradMode, gradFilter=gradFilter, gradScale=gradScale)
    push!(gradFilterSolutions, solutions)
    push!(gradFilterLosses, losses)
    push!(gradFilterParams, params)

    gif(anim, joinpath(@__DIR__, "translational_pendulum_x2_fps10_$(gradFilterNames[gf]).gif"), fps = 10)
end

##### PLOTTING #####

gradCOLORS = Colors.distinguishable_colors(length(gradFilters)+2, [RGB(1,1,1), RGB(0,0,0)])[3:end]
linewidth = 2

fig = startPlot(;ylabel=L"s~[m]", xlabel=L"t~[s]", legend=:bottomleft)
for gf in 1:length(gradFilters)

    loss = gradFilterLosses[gf]
    solutions = gradFilterSolutions[gf]

    #solution = solutions[round(Int, length(solutions)/2)]
    #Plots.plot!(fig, solution.t, collect(u[1] for u in solution.u); label="$(gradFilterNames[gf]) [2500]", color=gradCOLORS[gf], alpha=0.5, linewidth=linewidth)

    solution = solutions[end]
    Plots.plot!(fig, solution.t, collect(u[1] for u in solution.u); label="$(gradFilterNames[gf])", color=gradCOLORS[gf], linewidth=linewidth)
end
Plots.plot!(fig, refSol.t, collect(u[1] for u in refSol.u); label="ground truth", color=:black, style=:dash, linewidth=linewidth)
fig
savefig(fig, joinpath(@__DIR__, "translational_pendulum_x2_solution.svg"))
savefig(fig, joinpath(@__DIR__, "translational_pendulum_x2_solution.pdf"))

fig = startPlot(;ylabel=L"l_{SOL}~[m]", xlabel=L"steps", legend=:topright, yaxis=:log) # , ylims=(10.0^-1, 10^2.5))
#min1 = gradFilterLosses[1][end][2][1]
min2 = 0.5778 # gradFilterLosses[3][end][2][1]
#Plots.plot!(fig, [0,numSteps], [min1, min1]; label="local minima", color=:black, style=:dash)
#Plots.plot!(fig, [0,numSteps], [0, 0]; label="global minimum", color=:black, style=:dash)
for gf in 1:length(gradFilters)

    loss = gradFilterLosses[gf]
    steps = collect(l[1] for l in loss)
    losses = collect(l[2][1] for l in loss)

    Plots.plot!(fig, steps, losses; label=gradFilterNames[gf], color=gradCOLORS[gf], linewidth=linewidth)
end
Plots.plot!(fig, [0,numSteps], [min2, min2]; label="local minimum", color=:black, style=:dash, linewidth=linewidth)
fig
savefig(fig, joinpath(@__DIR__, "translational_pendulum_x2_convergence.svg"))
savefig(fig, joinpath(@__DIR__, "translational_pendulum_x2_convergence.pdf"))

fig = startPlot(;ylabel=L"\Re(\lambda_{w})~\left[\frac{1}{s}\right]", xlabel=L"steps", legend=:topright)
for gf in 1:length(gradFilters)

    loss = gradFilterLosses[gf]
    solutions = gradFilterSolutions[gf]
    params = gradFilterParams[gf]

    maxRes = []

    for i in 1:length(solutions)
        solution = solutions[i]
        param = params[i]

        maxRe = -Inf 
        _eigvalData = collect(eigvals(ForwardDiff.jacobian((u)->f(u, param, 0.0), u)) for u in solution.u)

        for eigvals in _eigvalData
            for eigval in eigvals 
                if real(eigval) > maxRe 
                    maxRe = real(eigval)
                end
            end
        end
        push!(maxRes, maxRe)
    end

    steps = collect(l[1] for l in loss)
    
    Plots.plot!(fig, steps, maxRes; label=gradFilterNames[gf], color=gradCOLORS[gf], linewidth=linewidth)
end
Plots.plot!(fig, [0,numSteps], [0, 0]; label="border stable", color=:black, style=:dash, linewidth=linewidth)
fig
savefig(fig, joinpath(@__DIR__, "translational_pendulum_x2_stability.svg"))
savefig(fig, joinpath(@__DIR__, "translational_pendulum_x2_stability.pdf"))