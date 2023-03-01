# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# This codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.

function build_model(flag_order, flag_deviation)
    # Parameters
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
    
    SS, PAR_SS = build_model_steadystate(ALPHA, BETA, DELTA, RHO, SIGMA, MUU, AA)

## Model processing
    # Store the model in a structure
    model = create_model(flag_order, flag_deviation,
        parameters, estimate,
        x, y, xp, yp,
        e, eta, f, SS, PAR_SS)

    return model

end