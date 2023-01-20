# using QuantEcon, Distributions
using SymPy, LinearAlgebra, Statistics  # Necessary
using Parameters # Cosmetic
using Plots # Plotting
using Random, BenchmarkTools # Timing

#v1.7+
# using MKL

#v1.7- 
BLAS.vendor() 
:mkl

include("solution_functions_4_0.jl")

## Model
    # Ajustes
        flag_order      =   2
        flag_deviation  =   true

    # Parameters
        @vars ALPHA BETA DELTA RHO SIGMA MUU
            parameters  =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]
            estimate    =   []

    # Variables
       @vars c cp k kp a ap
           x    =   [k; a]
           y    =   [c]
           xp   =   [kp; ap]
           yp   =   [cp]

   # Shock
        @vars epsilon
            e   =   [epsilon]
            eta =   Array([0.0; MUU])

    # Equilibrium conditions
            f1  =   c + kp - (1-DELTA) * k - a * k^ALPHA
            f2  =   c^(-SIGMA) - BETA * cp^(-SIGMA) * (ap * ALPHA * kp^(ALPHA-1) + 1 - DELTA)
            f3  =   log(ap) - RHO * log(a)

            f   =   [f1;f2;f3]

    # Steady state (Leave it as an empty vector for the code to find it)
            A   =   1.0
            K   =   ((1.0/BETA+DELTA-1.0)/ALPHA)^(1.0/(ALPHA-1.0))
            C   =   A * K^(ALPHA)-DELTA*K

            SS = [
                    log(K); # k
                    log(A); # a
                    log(C); # c
                    log(K); # kp
                    log(A); # ap
                    log(C); # cp
                ]

## Model processing
    # Store the model in a structure
        model = create_model(flag_order, flag_deviation,
                             parameters, estimate,
                             x, y, xp, yp,
                             e, eta, f, SS)

    # Create the reusable expresions for eta, SS and derivatives, and the functions to evaluate them efficiently

        ShockVAR_string = ShockVAR(model)
        eval(ShockVAR_string)

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
        SS      =   eval_SS(PAR)
        deriv   =   eval_deriv(PAR, SS)

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

## Timing
         @btime begin
             ALPHA   =   0.30
             BETA    =   0.95
             DELTA   =   1.00
             RHO     =   0.90
             SIGMA   =   2.00
             MUU     =   0.05
             PAR     =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]

             VAR     =   eval_ShockVAR(PAR)
             SS      =   eval_SS(PAR)
             deriv   =   eval_deriv(PAR, SS)

             sol_mat =   solve_perturbation(model, deriv, VAR)
         end
