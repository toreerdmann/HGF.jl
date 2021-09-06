
# HGF.jl

This package contains a re-implementation of the HGF update equations in Julia.

For inference, I implemented two methods:
* gradient-descent with re-starts
* MCMC (Metropolis-Hastings) using Turing.jl

Compared to the hgf-toolbox in MatLab, this package has highly reduced
functionality: only binary inputs and outputs. New response models can be added
easily though.


