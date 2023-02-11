# using QuantEcon, Distributions
using SymPy, LinearAlgebra, Statistics # Necessary
using Parameters # Cosmetic
using Random, BenchmarkTools # Timing

#v1.7+
# using MKL

#v1.7- 
BLAS.vendor() 
:mkl

include("solution_functions_5_0.jl")
include("smc_rwmh_neoclassical_5_0_v2.jl")

## Model
    # Ajustes
        flag_order      =   1
        flag_deviation  =   true

    # Parameters
        @vars ALPHA BETA DELTA RHO SIGMA MUU AA
            parameters  =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU; AA]
            estimate    =   [RHO; MUU]

    # Variables
       @vars k kp a ap c cp n np yy yyp r rp ii iip
           x    =   [k; a]
           y    =   [c; n; r; yy; ii]
           xp   =   [kp; ap]
           yp   =   [cp; np; rp; yyp; iip]

   # Shock
        @vars epsilon
            e   =   [epsilon]
            eta =   Array([0.0; MUU])

    # Equilibrium conditions
            f1  =   c + kp - (1-DELTA) * k - a * k^ALPHA * n^(1-ALPHA)
            f2  =   c^(-SIGMA) - BETA * cp^(-SIGMA) * (ap * ALPHA * kp^(ALPHA-1) * np^(1-ALPHA) + 1 - DELTA)
            f3  =   AA - c^(-SIGMA) * a * (1-ALPHA) * k^ALPHA * n^(-ALPHA) 
            f4  =   log(ap) - RHO * log(a)
            f5  =   r - a * ALPHA * k^(ALPHA-1) * n^(1-ALPHA)
            f6  =   yy - a * k^ALPHA * n^(1-ALPHA)
            f7  =   ii - (kp - (1-DELTA) * k)

            f   =   [f1;f2;f3;f4;f5;f6;f7]

    # Steady state (Leave it as an empty vector for the code to find it)
            A   =   1.0
            N   =   2/3
            K   =   (ALPHA/(1/BETA-1+DELTA))^(1/(1-ALPHA))*N
            C   =   A * K^(ALPHA) * N^(1-ALPHA) - DELTA*K
            R   =   A * ALPHA * K^(ALPHA-1.0) * N^(1-ALPHA)
            YY  =   A * K^ALPHA * N^(1-ALPHA)
            II  =   DELTA*K

            SS = [
                    log(K); # k
                    log(A); # a
                    log(C); # c
                    log(N);
                    log(R);
                    log(YY);
                    log(II);
                    log(K); # kp
                    log(A); # ap
                    log(C); # cp
                    log(N);
                    log(R);
                    log(YY);
                    log(II)
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

        println("Residuals: $SS_ER")

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
        @time estimation_results = smc_rwmh_neoclassical_4_0_v2(initial_para, data', model, PAR_SS, c)
