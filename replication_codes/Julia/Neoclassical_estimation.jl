include("solution_functions_7_0.jl")
using MKL

#########################################################################################################
# Estimation #
#########################################################################################################
flag_order = 1; flag_deviation = true; flag_SSsolver = false;

# Parameters
@vars ALPHA BETA DELTA RHO SIGMA MUU AA
parameters  =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU; AA]
estimate    =   [RHO; MUU]
npar = length(parameters)
ns = length(estimate)
estimation = [4;6]
priors = (prior_rho = (iv = .9 , lb = 0.0 , ub = 1.0, d = truncated(Normal(0.75, 0.25))),
        prior_muu = (iv = .05, lb = 0.0, ub = 1e6, d = InverseGamma(5.0, 0.25)))


# Variables
@vars k kp a ap c cp n np yy yyp r rp ii iip
x    =   [k; a]
y    =   [c; n; r; yy; ii]
xp   =   [kp; ap]
yp   =   [cp; np; rp; yyp; iip]
nx  = length(x)
ny  = length(y)
nvar   = nx + ny

variables = [x; y; xp; yp]

# Shock
@vars epsilon
e   =   [epsilon]
eta =   Array([0.0; MUU])
ne = length(e)

# Equilibrium conditions
f1  =   c + kp - (1-DELTA) * k - a * k^ALPHA * n^(1-ALPHA)
f2  =   c^(-SIGMA) - BETA * cp^(-SIGMA) * (ap * ALPHA * kp^(ALPHA-1) * np^(1-ALPHA) + 1 - DELTA)
f3  =   AA - c^(-SIGMA) * a * (1-ALPHA) * k^ALPHA * n^(-ALPHA) 
f4  =   log(ap) - RHO * log(a)
f5  =   r - a * ALPHA * k^(ALPHA-1) * n^(1-ALPHA)
f6  =   yy - a * k^ALPHA * n^(1-ALPHA)
f7  =   ii - (kp - (1-DELTA) * k)

f   =   [f1;f2;f3;f4;f5;f6;f7]

nf = length(f)

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

model = (parameters = parameters, estimate = estimate, estimation = estimation,
        npar = npar, ns = ns, priors = priors,
        x = x, y = y, xp = xp, yp = yp, variables = variables,
        nx = nx, ny = ny, nvar = nvar, 
        e = e, eta = eta,
        ne = ne,
        f = f,
        nf = nf,
        SS = SS, PAR_SS = PAR_SS,
        flag_order = flag_order, flag_deviation = flag_deviation, flag_SSsolver = flag_SSsolver)

process_model(model)

# Parametrization
ALPHA  =   0.30
BETA   =   0.95
DELTA  =   1.00
RHO    =   0.90
SIGMA  =   2.00
MUU    =   0.05

PAR     =   [ALPHA; BETA; DELTA; RHO; SIGMA; MUU]

eta     =   eval_ShockVAR(PAR)
PAR_SS  =   eval_PAR_SS(PAR)
SS      =   eval_SS(PAR_SS)
deriv   =   eval_deriv(PAR_SS, SS)

# @btime sol_mat = solve_model(model, deriv, eta)
sol_mat = solve_model(model, deriv, eta)

nsim = 200
# nsim = 500
simulation_logdev = simulate_model(model, sol_mat, nsim)
data = simulation_logdev[10001:10000+nsim,model.nx+1:model.nx+3]

c       = 0.1
npart   = 500#2^13          # of particles
# npart   = 2^13          # of particles
nphi    = 100 # 500         # of stage
lam     = 3#2.1             # bending coeff
# @time estimation_results = smc_rwmh_threads(model, data, PAR, eta, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam)
@time estimation_results = smc_rwmh(model, data, PAR, PAR_SS, eta, SS, deriv, sol_mat, c, npart, nphi, lam)