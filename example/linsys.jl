#
# Copyright (c) 2023 Tobias Thummerer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# 🚧 Disclaimer: This is draft code 🚧 #
# please check Project.toml for compatibility #

numStates = 2
include(joinpath(@__DIR__, "main.jl"))

# params
eta_start = 1e-2
eta_stop = 1e-3
eta_lambda = 2e-3 # 5e-4
numSteps = 5000 # 5000 # 7500
numRepetitions = 50
foldername = "results_linsys"

# consts 
tspan = (0.0, 10.0)
saveat = 0.0:0.25:10.0 # 0.0:0.1:10.0

# sys
c = (1 * 2*π)^2 / 1
p = [c, 0.05]
sys = (x) -> translational_pendulum(x, p)
u0 = [1.0, 0.0] 
reff = (x, p, t) -> sys(x) 

refODE = ODEFunction{false}(reff,tgrad=basic_tgrad);
refProb = ODEProblem{false}(refODE, u0, tspan);
refSol = solve(refProb)
plot(refSol)

# data and scales 
posData = collect(u[1] for u in refSol.u)
velData = collect(u[2] for u in refSol.u)
accData = collect(refSol(t, Val{1})[2] for t in refSol.t)
posScale = 1/max(abs.(posData)...)
velScale = 1/max(abs.(velData)...)
accScale = 1/max(abs.(accData)...)

refSol = solve(refProb;saveat=saveat)
posData = collect(u[1] for u in refSol.u)
velData = collect(u[2] for u in refSol.u)
accData = collect(refSol(t, Val{1})[2] for t in refSol.t)
u0_test = refSol.u[end]

# data gathering
eigvalData = collect(_eigen(ForwardDiff.jacobian(sys, u))[1] for u in refSol.u)
isLinear = eigvalData[1] == eigvalData[end]
stiffnessData = prepareStiffness(eigvalData)
eigsPairs = prepareEigPairs(eigvalData)

#              ["SOL", "FRQ", "DMP", "STB", "OSC", "STF"]
gradFilters = [[   1,     0,     0,     0,     0,     0],
               [   1,     0,     0,     1,     0,     0],
               [   1,     0,     0,     0,     0,     1],
               [   1,     0,     0,     1,     1,     0],
               [   1,     1,     1,     0,     1,     0],
               [   1,     1,     1,     1,     1,     0],
               [   1,     1,     1,     1,     1,     1]]

gradFilterNames = getGradFilterNames(gradFilters)

while (experiment = startExperiment(foldername, numRepetitions)) != nothing

    # setup repetition specific stuff 
    net = Chain(preprocess,
        Dense(numStates, layerwidth, tanh),
        Dense(layerwidth, Int(numStates/2)),
        postprocess)
    p_orig, st = Lux.setup(rng, net)
    p_orig = Float64.(ComponentArray(p_orig))

    for gf in 1:length(gradFilters)

        push!(experiment) # new run! 

        _p_orig = deepcopy(p_orig)
        gradFilter = gradFilters[gf]

        solver = Tsit5()
        neuralODE = NeuralODE(net, tspan, solver; saveat=saveat)

        _loss_parts = p -> loss_parts(neuralODE, u0, p, st, gradFilter)
        _loss = p -> loss(neuralODE, u0, p, st, loss_parts, gradFilter, gradScale)
        _loss_test = p -> loss_test(neuralODE, u0_test, p, st)
        _callback = (state, val) -> callback(state, val, numSteps, experiment, _loss, _loss_parts, _loss_test, gradScale, gradFilter, eta_start, eta_stop, eta_lambda)

        loss_before = _loss(_p_orig)
        _loss_parts(_p_orig)
        @info "Loss before: $(loss_before)"

        optf = Optimization.OptimizationFunction((x, p) -> _loss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, _p_orig)
        result = Optimization.solve(optprob, 
                OptimizationOptimisers.Adam(eta_start); 
                callback=_callback, 
                maxiters=numSteps)
        optprob = remake(optprob; u0=result.u)

        loss_after = _loss(optprob.u0)
        @info "Loss after: $(loss_after)"
    end

    finishExperiment(foldername, experiment)
end

checkLastExperiment(foldername)

experiments = loadExperiments(foldername)

##### PLOTTING #####

gr()
plotConvergence("translational_pendulum_x2_convergence.pdf", experiments)
plotMedian("translational_pendulum_x2_solution.pdf", experiments, u0)
plotStability("translational_pendulum_x2_stability.pdf", experiments)
latexResults("linsys", experiments)

# plotlyjs()
# begin 
#     net, _, st, solver, neuralODE = setup()
#     for i in 1:7
#         display(plotContributions(neuralODE, experiments, i, u0, st))
#     end
# end