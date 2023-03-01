# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# This codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.


function build_model_steadystate(ALPHA, BETA, DELTA, RHO, SIGMA, MUU, AA)
    

    
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



    
 return SS, PAR_SS

end