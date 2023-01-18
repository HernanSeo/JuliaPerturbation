# using QuantEcon, Distributions
using SymPy, LinearAlgebra, Statistics # Necessary
using Parameters # Cosmetic
using Random, BenchmarkTools # Timing

BLAS.vendor()
:mkl

include("solution_functions_4_0.jl")
include("smc_rwmh_neoclassical_4_0_v2.jl")

## Model
    # Ajustes
        flag_order      =   1
        flag_deviation  =   true

    # Parameters
        @vars ALPHA BETA DELTA RHO SIGMA MUU
            parameters  =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]
            estimate    =   [RHO; MUU]

    # Variables
       @vars c cp k kp a ap yy yyp r rp ii iip
           x    =   [k; a]
           y    =   [c; r; yy; ii]
           xp   =   [kp; ap]
           yp   =   [cp; rp; yyp; iip]

   # Shock
        @vars epsilon
            e   =   [epsilon]
            eta =   Array([0.0; MUU])

    # Equilibrium conditions
            f1  =   c + kp - (1-DELTA) * k - a * k^ALPHA
            f2  =   c^(-SIGMA) - BETA * cp^(-SIGMA) * (ap * ALPHA * kp^(ALPHA-1) + 1 - DELTA)
            f3  =   log(ap) - RHO * log(a)
            f4  =   r - a * ALPHA * k^(ALPHA-1)
            f5  =   yy - a * k^ALPHA
            f6  =   ii - (kp - (1-DELTA) * k)

            f   =   [f1;f2;f3;f4;f5;f6]

    # Steady state (Leave it as an empty vector for the code to find it)
            A   =   1.0
            K   =   ((1.0/BETA+DELTA-1.0)/ALPHA)^(1.0/(ALPHA-1.0))
            C   =   A * K^(ALPHA)-DELTA*K
            R   =   A * ALPHA*K^(ALPHA-1.0)
            YY  =   A * K^ALPHA
            II  =   DELTA*K

            SS = [
                    log(K); # k
                    log(A); # a
                    log(C); # c
                    log(R);
                    log(YY);
                    log(II);
                    log(K); # kp
                    log(A); # ap
                    log(C); # cp
                    log(R);
                    log(YY);
                    log(II)
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
        ALPHA  =  0.3
        BETA   =  0.95
        DELTA  =  0.01
        RHO    =  0.90
        SIGMA  =  2.0
        MUU    =  0.05

        PAR     =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]

        println("Evaluate derivatives")
        VAR = eval_ShockVAR(PAR)
        SS =  eval_SS(PAR)
        deriv = eval_deriv(PAR, SS)

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
        data = Simulation_logdev[model.nx+1:model.nx+3,TS-499:TS] # las observables se ordenan al final del vector de simulations

## Estimation
        initial_para=[0.9; 0.15]
        c=0.1
        @time estimation_results = smc_rwmh_neoclassical_4_0_v2(initial_para, data', model, PAR, c)
