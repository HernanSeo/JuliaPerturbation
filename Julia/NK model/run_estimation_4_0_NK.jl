# using QuantEcon, Distributions
using SymPy, LinearAlgebra, Statistics # Necessary
using Parameters # Cosmetic
using Random, BenchmarkTools # Timing

BLAS.vendor()
:mkl

include("solution_functions_4_0.jl")
include("smc_rwmh_nk_4_0_v2.jl")

## Model
    # Ajustes
        flag_order = 1
        flag_deviation = true

    # Parameters
        @vars THETA BETTA GAMMA DELTA ETA ALPHA XI RHO PHI_R PHI_PPI PHI_Y PPS YSS RRS SIGMAA
            parameters  =   [THETA; BETTA; GAMMA; DELTA; ETA; ALPHA; XI; RHO; PHI_R; PHI_PPI; PHI_Y; PPS; YSS; RRS; SIGMAA]
            estimate    =   [RHO; SIGMAA]

    # Variables
       @vars c cp h hp l lp w wp mc mcp s sp ph php ppi ppip rt rtp x1 x1p x2 x2p br brp brb brbp z zp k kp ii iip u up
           x = [s; z; brb; k]
           y = [c; h; ppi; rt; br; w; l; mc; ph; x1; x2; ii; u]
           xp = [sp; zp; brbp; kp]
           yp = [cp; hp; ppip; rtp; brp; wp; lp; mcp; php; x1p; x2p; iip; up]

   # Shock
        @vars epsilon
            e = [epsilon]
            eta = Array([0.0; SIGMAA; 0.0; 0.0])

    # Equilibrium conditions
            f1 = -kp + (1-DELTA)*k + ii
            f2  = l - (c*(1-h)^GAMMA)^(-XI) * (1-h)^GAMMA
            f3  = BETTA*lp/ppip - l*rt
            f4  = 1/br - rt
            f5  = -sp + (1-THETA)*(ph^(-ETA)) + THETA*(ppi^ETA)*s
            f6  = z*h^(1-ALPHA)*k^(ALPHA) - sp*(c+ii)
            f7  = -1 + (1-THETA)*(ph^(1-ETA)) + THETA*(ppi^(ETA-1))
            f8  = -x1 + (ph^(1-ETA))*c*((ETA-1)/ETA) + THETA*rt*((ph/php)^(1-ETA))*(ppip^(ETA))*x1p
            f9  = -x2 + (ph^(-ETA))*c*mc + THETA*rt*((ph/php)^(-ETA))*(ppip^(ETA+1))*x2p
            f10 = x2-x1
            f11 = w - mc*z*(1-ALPHA)*h^(-ALPHA)*k^(ALPHA)
            f12 = w - ((c*(1-h)^GAMMA)^(-XI)*GAMMA*(1-h)^(GAMMA-1)*c)/((c*(1-h)^GAMMA)^(-XI)*(1-h)^GAMMA)
            f13a = u - mc*z*ALPHA*h^(1-ALPHA)*k^(ALPHA-1)
            f13b = l - BETTA*(up+1-DELTA)*lp
            f14a = log(br/RRS) - PHI_R*log(brb/RRS) - PHI_PPI*log(ppi/PPS) - PHI_Y*log((sp*(c+ii))/YSS)
            f14b = brbp - br
            f15 = log(zp) - RHO * log(z)

            f = [f1;f2;f3;f4;f5;f6;f7;f8;f9;f10;f11;f12;f13a;f13b;f14a;f14b;f15]

    # Steady state (Leave it as an empty vector for the code to find it)
            Z           =       1
            R           =       1/RRS
            RR          =       RRS
            PH          =       ((1-THETA*PPS^(ETA-1))/(1-THETA))^(1/(1-ETA))
            PPI         =       PPS
            S           =       (1-THETA)*PH^(-ETA)/(1-THETA*PPS^(ETA))
            MC          =       ((1-THETA*RRS^(-1)*PPS^(ETA+1))*((ETA-1)/ETA)*PH)/(1-THETA*RRS^(-1)*PPS^(ETA))
            U           =       1/BETTA - 1 + DELTA
            H           =       ((1-ALPHA)*MC*Z*((ALPHA*Z*MC)/U)^(ALPHA/(1-ALPHA)))/((1-ALPHA)*MC*Z*((ALPHA*Z*MC)/U)^(ALPHA/(1-ALPHA)) + (GAMMA*Z*((ALPHA*Z*MC)/U)^(ALPHA/(1-ALPHA)) - S*DELTA*((ALPHA*Z*MC)/U)^(1/(1-ALPHA)))/S);
            K           =       ((ALPHA*Z*MC)/U)^(1/(1-ALPHA))*H
            W           =       (1-ALPHA)*MC*Z*(K/H)^ALPHA
            Y           =       Z*H^(1-ALPHA)*K^ALPHA
            YSS         =       Y
            C           =       (Y - S*DELTA*K)/S
            II          =       DELTA*K
            X1          =       (PH^(1-ETA)*Y*S^(-1)*((ETA-1)/ETA))/(1-THETA*RRS^(-1)*PPS^ETA)
            X2          =       X1
            LAM         =       C^(-XI)*(1-H)^(GAMMA*(1-XI))
            SS = [
                    log(S);
                    log(Z);
                    log(RR);
                    log(K);
                    log(C);
                    log(H);
                    log(PPI);
                    log(R);
                    log(RR);
                    log(W);
                    log(LAM);
                    log(MC);
                    log(PH);
                    log(X1);
                    log(X2);
                    log(II);
                    log(U);
                    log(S);
                    log(Z);
                    log(RR);
                    log(K);
                    log(C);
                    log(H);
                    log(PPI);
                    log(R);
                    log(RR);
                    log(W);
                    log(LAM);
                    log(MC);
                    log(PH);
                    log(X1);
                    log(X2);
                    log(II);
                    log(U);
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

        SS_string  = steadystate(model)
        eval(SS_string)

        deriv_string = derivatives(model)
        eval(deriv_string)

## Solution
     # Parametrization
             THETA      =       0.80
             BETTA      =       1.04^(-1/4)
             GAMMA      =       3.6133
             DELTA      =       0.01
             ETA        =       5.00
             ALPHA      =       0.30
             XI         =       2.00
             RHO        =       0.8556
             Z          =       1.00
             PHI_PPI    =       3.00
             PHI_Y      =       0.01
             PHI_R      =       0.80
             PPS        =       1.042^(1/4)
             RRS        =       PPS/BETTA
             SIGMAA     =       0.05

             Z          =       1.00
             R          =       1.00/RRS
             RR         =       RRS
             PH         =       ((1.00-THETA*PPS^(ETA-1.00))/(1.00-THETA))^(1.00/(1.00-ETA))
             PPI        =       PPS
             S          =       (1.00-THETA)*PH^(-ETA)/(1.00-THETA*PPS^(ETA))
             MC         =       ((1.00-THETA*RRS^(-1.00)*PPS^(ETA+1.00))*((ETA-1.00)/ETA)*PH)/(1.00-THETA*RRS^(-1.00)*PPS^(ETA))
             U          =       1.00/BETTA - 1.00 + DELTA
             H          =       ((1.00-ALPHA)*MC*Z*((ALPHA*Z*MC)/U)^(ALPHA/(1.00-ALPHA)))/((1.00-ALPHA)*MC*Z*((ALPHA*Z*MC)/U)^(ALPHA/(1.00-ALPHA)) + (GAMMA*Z*((ALPHA*Z*MC)/U)^(ALPHA/(1.00-ALPHA)) - S*DELTA*((ALPHA*Z*MC)/U)^(1.00/(1.00-ALPHA)))/S)
             K          =       ((ALPHA*Z*MC)/U)^(1.00/(1.00-ALPHA))*H
             W          =       (1.00-ALPHA)*MC*Z*(K/H)^ALPHA
             YSS        =       Z*H^(1.00-ALPHA)*K^ALPHA

        PAR = [THETA; BETTA; GAMMA; DELTA; ETA; ALPHA; XI; RHO; PHI_R; PHI_PPI; PHI_Y; PPS; YSS; RRS; SIGMAA]

        println("Evaluate derivatives")
        VAR = eval_ShockVAR(PAR)
        SS =  eval_SS(PAR)
        deriv = eval_deriv(PAR, SS)

     # Solve the model given the current parametrization
        println("Solving first order")
        sol_mat = solve_perturbation(model, deriv, VAR)

## Sample for estimation
        TS = 1000
        Simulation_logdev = simulation_dsge(model, sol_mat, SS, VAR, TS, zeros(model.nx), true, false)
        data = Simulation_logdev[model.nx+1:model.nx+3,TS-199:TS] # las observables se ordenan al final del vector de simulations

## Estimation
        initial_para = [0.9; 0.05]
        c=0.1
        @time estimation_results = smc_rwmh_nk_4_0_v2(initial_para, data', model, PAR, c)
