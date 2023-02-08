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
include("smc_rwmh_neoclassical_5_0_v2.jl")

## Model
    # Ajustes
    flag_order      =   1
    flag_deviation  =   true

## Model
    # Ajustes
    flag_order      =   2
    flag_deviation  =   true

# Parameters
    @vars ALPHA BETA DELTA RHO SIGMA MUU AA
        parameters  =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU; AA]
        estimate    =   []

# Variables
    # States
    @vars k kp a ap c cp  n np
        x    =   [k; a]
        y    =   [c; n]
        xp   =   [kp; ap]
        yp   =   [cp; np]

    # Shock
    @vars epsilon
        e   =   [epsilon]
        eta =   Array([0.0; MUU])

# Equilibrium conditions
   f1  =   c + kp - (1-DELTA) * k - a * k^ALPHA * n^(1-ALPHA)
   f2  =   c^(-SIGMA) - BETA * cp^(-SIGMA) * (ap * ALPHA * kp^(ALPHA-1) * n^(1-ALPHA) + 1 - DELTA)
   f3  =   AA - c^(-SIGMA) * a * (1-ALPHA) * k^ALPHA * n^(-ALPHA)  
   f4  =   log(ap) - RHO * log(a)

   f   =   [f1;f2;f3;f4]

# Steady state (Leave it as an empty vector for the code to find it)
   A   =   1.0
   N   =   2/3
   K   =   (ALPHA/(1/BETA-1+DELTA))^(1/(1-ALPHA))*N
   C   =   A * K^ALPHA * N^(1-ALPHA) - DELTA*K 
   
   SS = [
           log(K); # k
           log(A); # a
           log(C); # c
           log(N); # n
           log(K); # kp
           log(A); # ap
           log(C); # cp
           log(N)  # np
       ]
# Parameters to adjust
    AA = C^(-SIGMA) * (1-ALPHA)*A*K^(ALPHA)*N^(-ALPHA)
    PAR_SS = [ALPHA; BETA; DELTA; RHO; SIGMA; MUU; AA]

## Model processing
    # Store the model in a structure
        model = create_model(flag_order, flag_deviation,
        parameters, estimate,
        x, y, xp, yp,
        e, eta, f, SS, PAR_SS)

    # Create the reusable expresions for eta, SS and derivatives, and the functions to evaluate them efficiently
        ShockVAR_string = ShockVAR(model)
        eval(ShockVAR_string)

        PAR_SS_string   = adjustpar(model)
        eval(PAR_SS_string)
        
        SS_string       = steadystate(model)
        eval(SS_string)

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
        deriv   =   eval_deriv(PAR_SS, SS)

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

