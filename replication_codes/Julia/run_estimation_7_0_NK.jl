# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# These codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.

#v1.7+
using MKL

#v1.7- 
#BLAS.vendor() 
#:mkl

include("solution_functions_7_0.jl")

## Model
    # Adjustment
        flag_order      = 1
        flag_deviation  = true
        flag_SSsolver   = false

    # Parameters
        @vars THETA BETTA GAMMA DELTA ETA ALPHA XI RHO PHI_R PHI_PPI PHI_Y PPS YSS RRS SIGMAA
        parameters  =   [THETA; BETTA; GAMMA; DELTA; ETA; ALPHA; XI; RHO; PHI_R; PHI_PPI; PHI_Y; PPS; YSS; RRS; SIGMAA]
        estimate    =   [RHO; SIGMAA]
        position = [8; 15]
        priors = (prior_rho = (iv = .9 , lb = 0.0 , ub = 1.0, d = truncated(Normal(0.75, 0.25),0.0,1.0)),
                  prior_sigma = (iv = .05, lb = 0.0, ub = 1e6, d = InverseGamma(5.0, 0.25)))

    # Variables
        @vars c cp h hp l lp w wp mc mcp s sp ph php ppi ppip rt rtp x1 x1p x2 x2p br brp brb brbp z zp k kp ii iip u up
        x = [s; z; brb; k]
        y = [c; h; ppi; rt; br; w; l; mc; ph; x1; x2; ii; u]
        xp = [sp; zp; brbp; kp]
        yp = [cp; hp; ppip; rtp; brp; wp; lp; mcp; php; x1p; x2p; iip; up]
        variables = [x; y; xp; yp]

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

    # Steady state
        # Values
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
        # Vector
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

        PAR_SS = [THETA; BETTA; GAMMA; DELTA; ETA; ALPHA; XI; RHO; PHI_R; PHI_PPI; PHI_Y; PPS; YSS; RRS; SIGMAA]

    # Procesing the model (no adjustment needed) 
        model = (parameters = parameters, estimate = estimate, estimation = position,
                    npar = length(parameters), ns = length(estimate), priors = priors,
                    x = x, y = y, xp = xp, yp = yp, variables = variables,
                    nx = length(x), ny = length(y), nvar = length(x) + length(y),
                    e = e, eta = eta,
                    ne = length(e),
                    f = f,
                    nf = length(f),
                    SS = SS, PAR_SS = PAR_SS,
                    flag_order = flag_order, flag_deviation = flag_deviation, flag_SSsolver = flag_SSsolver)

        process_model(model)

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

    # Solution functions (no adjustment needed)    
        eta     =   eval_ShockVAR(PAR)
        PAR_SS  =   eval_PAR_SS(PAR)
        SS      =   eval_SS(PAR_SS)
        deriv   =   eval_deriv(PAR_SS, SS)
        
        # @btime sol_mat = solve_model(model, deriv, eta)
        sol_mat = solve_model(model, deriv, eta)

## Estimation
    # Simulate a sample for estimation         
        nsim = 200
        # nsim = 500
        discard = 10000
        flag_IR = false
        flag_logdev = true
        simulation_logdev = simulate_model(model, sol_mat, discard+nsim, eta, SS, flag_IR, flag_logdev)
        data = simulation_logdev[discard+1:discard+nsim,model.nx+1:model.nx+3]
    
    # Estimation
        c       = 0.1
        npart   = 500           # of particles
        # npart   = 2^13        # of particles
        nphi    = 100           # of stage
        lam     = 3             # bending coeff
        @time estimation_results = smc_rwmh(model, data, PAR, PAR_SS, eta, SS, deriv, sol_mat, c, npart, nphi, lam)