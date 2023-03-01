# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# This codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.

# using QuantEcon, Distributions
using SymPy, LinearAlgebra, Statistics, Plots # Necessary
using Parameters # Cosmetic
using Random, BenchmarkTools # Timing

#v1.7+
using MKL

#v1.7- 
# BLAS.vendor() 
# :mkl

include("solution_functions_5_0.jl")
# include("smc_rwmh_neoclassical_5_0_v2.jl")

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
        ALPHA  =   0.30
        BETA   =   0.95
        DELTA  =   1.00
        RHO    =   0.90
        SIGMA  =   2.00
        MUU    =   0.05
        
        PAR     =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]

    # Evaluate the expresions of the SS and derivatives given the current parametrization
        VAR     =   eval_ShockVAR(PAR)
        PAR_SS  =   eval_PAR_SS(PAR)
        SS      =   eval_SS(PAR_SS)
        SS_ER   =   eval_SS_error(PAR_SS, SS)
        deriv   =   eval_deriv(PAR_SS, SS)
        println("Residuals: $SS_ER")

    # Solve the model given the current parametrization
        sol_mat =   solve_perturbation(model, deriv, VAR)
        println("Solved")

## Simulation
    # Simulation
        T_S = 100
        IS_S = zeros(model.nx)
        Simulation = simulation_dsge(model, sol_mat, SS, VAR, T_S, IS_S, true, false)
        plot(Simulation')

    # Impulse-response functions
        T_IR = 20
        IS_IR = [0.0; MUU]
        ImpRes = simulation_dsge(model, sol_mat, SS, VAR, T_IR,IS_IR , false, false)
        plot(ImpRes')

