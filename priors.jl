# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# These codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.


using SymPy, LinearAlgebra, Statistics # Necessary
using Parameters # Cosmetic
using Random, BenchmarkTools # Timing

# this file contains the function for the initial draw and the logpriors

function initial_draw(npart, nphi, npara)

    # drawing initial values from the prior distributions
    parasim = zeros(nphi, npart, npara) # parameter draws

# truncated normal
mean1 = 0.75
std1 = 0.25
d1 = Normal(mean1, std1) #Normal{Float64}(μ=0.16, σ=0.05)
td1 = truncated(d1, 0.0, 1.0)
parasim[1, :, 1] = rand(td1, npart)

# inverse gamma
shape2 = 5.0
scale2 = 0.25
d2 = InverseGamma(shape2, scale2) # Normal{Float64}(μ=0.16, σ=0.05)
parasim[1, :, 2] = rand(d2, npart)


    return parasim

end


function logpriors(p0)
    nparams=size(p0)
    idx=NaN*zeros(nparams)

    mean1 = 0.75
    std1 = 0.25
    if p0[1]<0.0 || p0[1]>1.0
        idx[1]=0.0
    else
        idx[1]=pdf(Normal(mean1,std1), p0[1])
    end

    shape2=5.0
    scale2=0.25
    if p0[2]<=0.0
        idx[2]=0.0
    else
        idx[2] = pdf(InverseGamma(shape2, scale2),p0[2])
    end
    lnp=sum(log.(idx))
    return lnp
end
