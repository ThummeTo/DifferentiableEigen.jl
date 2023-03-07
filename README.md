# DifferentiableEigen.jl
[![Run Tests](https://github.com/ThummeTo/DifferentiableEigen.jl/actions/workflows/Test.yml/badge.svg)](https://github.com/ThummeTo/DifferentiableEigen.jl/actions/workflows/Test.yml)
[![Coverage](https://codecov.io/gh/ThummeTo/DifferentiableEigen.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ThummeTo/DifferentiableEigen.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

## What is DifferentiableEigen.jl?
The current implementation of `LinearAlgebra.eigen` does not support sensitivities. 
This package adds a new function `eigen`, that wraps the original function, but returns an array of reals instead of complex numbers (this is necessary, because some AD-frameworks do not support complex numbers). 
This `eigen` function is differentiable by every AD-framework with support for *ChainRulesCore.jl* and *ForwardDiff.jl*.

## How can I use DifferentiableEigen.jl?
1\. Open a Julia-REPL, switch to package mode using `]`, activate your preferred environment.

2\. Install [*DifferentiableEigen.jl*](https://github.com/ThummeTo/DifferentiableEigen.jl):
```julia-repl
(@v1.6) pkg> add DifferentiableEigen
```

3\. If you want to check that everything works correctly, you can run the tests bundled with [*DifferentiableEigen.jl*](https://github.com/ThummeTo/DifferentiableEigen.jl):
```julia-repl
(@v1.6) pkg> test DifferentiableEigen
```

## How does it work? 
```julia
import DifferentiableEigen
import LinearAlgebra
import ForwardDiff

A = rand(3,3)   # Random matrix 3x3 

eigvals, eigvecs = LinearAlgebra.eigen(A)   # This is the default eigen-function in Julia. Note, that eigenvalues and -vectors are complex numbers.
jac = ForwardDiff.jacobian((A) -> LinearAlgebra.eigen(A)[1], A)   # That doesn't work!

eigvals, eigvecs = DifferentiableEigen.eigen(A)   # This is the differentiable eigen-function. Note, that eigenvalues and -vectors are not complex numbers, but real arrays!  
jac = ForwardDiff.jacobian((A) -> DifferentiableEigen.eigen(A)[1], A)   # That does work! eigenvalue- and eigenvector-sensitvities
```

## Acknowledgement
This package was motivated by this [discourse thread](https://discourse.julialang.org/t/native-eigenvals-for-differentiable-programming/27126). 
For now, there is no other (known) ready to use solution for differentiable eigenvalues and -vectors. 
If this changes, please feel free to open a PR or discussion.

The sensitivity formulas are picked from:

Michael B. Giles. 2008. **An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation.** [PDF](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)

## How to cite? Related publications?
Tobias Thummerer and Lars Mikelsons. 2023. **Eigen-informed NeuralODEs: Dealing with stability and convergence issues of NeuralODEs.** ArXiv. [PDF](https://www.researchgate.net/publication/368364260_Eigen-informed_NeuralODEs_Dealing_with_stability_and_convergence_issues_of_NeuralODEs)

