# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# These codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.

# using QuantEcon, Distributions
using SymPy, LinearAlgebra, Statistics # Necessary
using Parameters # Cosmetic
using Random, BenchmarkTools # Timing
using Distributed

#v1.7+
using MKL

#v1.7- 
#BLAS.vendor() 
#:mkl

include("solution_functions_5_0.jl")
include("smc_rwmh_neoclassical_5_0_v2.jl")

# this file includes the Model
include("build_model.jl")

# this file includes the solution for the steady state
include("build_model_steadystate.jl")

include("priors.jl")

## Model
    # Ajustes
        flag_order      =   1
        flag_deviation  =   true


        model = build_model(flag_order, flag_deviation)

## Model processing

    # Create the reusable expresions for eta, SS and derivatives, and the functions to evaluate them efficiently
        ShockVAR_string = ShockVAR(model)
        eval(ShockVAR_string)

        PAR_SS_string   = adjustpar(model)
        eval(PAR_SS_string)
        
        SS_string       = steadystate(model)
        eval(SS_string)

        SS_error_string = ss_error(model)
        eval(SS_error_string)

        deriv_string    = derivatives(model)
        eval(deriv_string)

## Solution
     # Parametrization
        ALPHA  =  0.3
        BETA   =  0.95
        DELTA  =  0.01
        RHO    =  0.90
        SIGMA  =  2.0
        MUU    =  0.05

        PAR     =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]

        # Evaluate the expresions of the SS and derivatives given the current parametrization
        VAR     =   eval_ShockVAR(PAR)
        PAR_SS  =   eval_PAR_SS(PAR)
        SS      =   eval_SS(PAR_SS)
        SS_err  =   eval_SS_error(PAR_SS, SS)
        deriv   =   eval_deriv(PAR_SS, SS)

        println("Residuals: $SS_err")

     # Solve the model given the current parametrization
        println("Solving first order")
        sol_mat = solve_perturbation(model, deriv, VAR)

     # Simulate a sample for estimation
        TS = 1000
        Simulation_logdev = simulation_dsge(model, sol_mat, SS, VAR, TS, zeros(model.nx), true, false)

        # plot(Simulation', layout = n, title = Vars, label = Vars)

        # Means = mean(Simulation, dims = 2)
        # DesSt = std(Simulation, dims = 2)

        # for the estimation I will use the log deviations
        #data = Simulation_logdev[nx+1:nx+ny,TS-120:TS] # las observables se ordenan al final del vector de simulations
        data = Simulation_logdev[model.nx+1:model.nx+3,TS-199:TS] # las observables se ordenan al final del vector de simulations

## Estimation
        initial_para=[0.9; 0.15]
        c=0.1
        @time estimation_results = smc_rwmh_neoclassical_4_0_v2(initial_para, data', model, PAR_SS, c)
