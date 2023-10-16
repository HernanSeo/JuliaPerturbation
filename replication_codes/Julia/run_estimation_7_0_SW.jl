include("solution_functions_7_0.jl")
using MKL

#########################################################################################################
# Estimation #
#########################################################################################################
flag_order = 1; flag_deviation = false; flag_SSsolver = true;

# Parameters
@vars ctou clandaw cg curvp curvw
@vars cgamma cbeta cpie 
@vars ctrend constebeta constepinf constelab calfa csigma cfc cgy
@vars csadjcost chabb cprobw csigl cprobp cindw cindp czcap crpi crr cry crdy crhoa crhob crhog crhoqs crhoms crhopinf crhow cmap cmaw
@vars clandap cbetabar cr crk cw cikbar cik clk cky ciy ccy crkky cwhlc cwly
@vars sda sdb sdq sdm sdg sdp sdw
parameters = [ctou; clandaw; cg; curvp; curvw;
            cgamma; cbeta; cpie; 
            ctrend; constebeta; constepinf; constelab; calfa; csigma; cfc; cgy; 
            csadjcost; chabb; cprobw; csigl; cprobp; cindw; cindp; czcap; crpi; crr; cry; crdy; 
            crhoa; crhob; crhog; crhoqs; crhoms; crhopinf; crhow; cmap; cmaw;
            clandap; cbetabar; cr; crk; cw; cikbar; cik; clk; cky; ciy; ccy; crkky; cwhlc; cwly;
            sda; sdb; sdq; sdm; sdg; sdp; sdw]
# estimate = [crhoa; crhob; crhog; crhoqs; crhoms; crhopinf; crhow;
#             sda; sdb; sdq; sdm; sdg; sdp; sdw]
estimate = [crhoa; sda]
# estimate = [crhoa; crhob; crhog]
npar = length(parameters)
ns = length(estimate)
# estimation = [collect(29:35); collect(52:58)]
estimation = [29; 52]
# priors = (  prior_crhoa = (iv = .9676 , lb = .01, ub = .9999, d = Normal(0.9, 0.05)),
#             prior_crhob = (iv = .2703, lb = .01, ub = .9999, d = Normal(0.2, 0.05)),
#             prior_crhog = (iv = .9930, lb = .01, ub = .9999, d = Normal(0.9, 0.05)),
#             prior_crhoqs = (iv = .5724, lb = .01, ub = .9999, d = Normal(0.75, 0.05)),
#             prior_crhoms = (iv = .3, lb = .01, ub = .9999, d = Normal(0.2, 0.05)),
#             prior_crhopinf = (iv = .8692, lb = .01, ub = .9999, d = Normal(0.9, 0.05)),
#             prior_crhow = (iv = .9546, lb = .001, ub = .9999, d = Normal(0.9, 0.05)),
#             prior_sda = (iv = 0.4618, lb = 1e-6, ub = 3.0, d = InverseGamma(5.0, 0.25)),
#             prior_sdb  = (iv = 0.1818513, lb = 1e-6, ub = 5, d = InverseGamma(5.0, 0.25)),
#             prior_sdg  = (iv = 0.6090, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
#             prior_sdqs  = (iv = 0.46017, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
#             prior_sdm  = (iv = 0.2397, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
#             prior_sdpinf  = (iv = 0.1455, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
#             prior_sdw  = (iv =0.2089, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)))  
priors = (  prior_crhoa = (iv = .9676 , lb = .01, ub = .999, d = truncated(Normal(0.9, 0.05),0.0,1.0)),
            # prior_crhob = (iv = .2703, lb = .01, ub = .999, d = Normal(0.2, 0.05)),
            # prior_crhog = (iv = .9930, lb = .01, ub = .999, d = Normal(0.9, 0.05)),
            # prior_crhoqs = (iv = .5724, lb = .01, ub = .999, d = Normal(0.5, 0.05)),
            # prior_crhoms = (iv = .3, lb = .01, ub = .999, d = Normal(0.2, 0.05)),
            # prior_crhopinf = (iv = .8692, lb = .01, ub = .999, d = Normal(0.9, 0.05)),
            # prior_crhow = (iv = .9546, lb = .001, ub = .999, d = Normal(0.9, 0.05)))
            prior_sda = (iv = 0.4618, lb = 1e-6, ub = 3.0, d = InverseGamma(5.0, 0.25)))
            # prior_sdb  = (iv = 0.1818513, lb = 1e-6, ub = 5, d = InverseGamma(5.0, 0.25)),
            # prior_sdg  = (iv = 0.6090, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
            # prior_sdqs  = (iv = 0.46017, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
            # prior_sdm  = (iv = 0.2397, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
            # prior_sdpinf  = (iv = 0.1455, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)),
            # prior_sdw  = (iv =0.2089, lb = 1e-6, ub = 3, d = InverseGamma(5.0, 0.25)))  
    
    
# Variables
@vars a b qs ms g spinf sw
@vars epinfma ewma
@vars invefl ccfl yfl
@vars invel cl pinfl wl yl rl 
@vars kpf kp
@vars a_p b_p qs_p ms_p g_p spinf_p sw_p
@vars epinfma_p ewma_p
@vars invefl_p ccfl_p yfl_p
@vars invel_p cl_p pinfl_p wl_p yl_p rl_p
@vars kpf_p kp_p
@vars rkf wf zcapf labf kkf invef pkf rrf yf ccf
@vars mc rk w zcap lab kk inve pk r pinf yy c
@vars rkf_p wf_p zcapf_p labf_p kkf_p invef_p pkf_p rrf_p yf_p ccf_p   
@vars mc_p rk_p w_p zcap_p lab_p kk_p inve_p pk_p r_p pinf_p y_p c_p

x = [a; b; qs; ms; g; spinf; sw;
    epinfma; ewma;
    invefl; ccfl; yfl;
    invel; cl; pinfl; wl; yl; rl;
    kpf; kp]
y = [rkf; wf; zcapf; labf; kkf; invef; pkf; rrf; yf; ccf;
    mc; rk; w; zcap; lab; kk; inve; pk; r; pinf; yy; c]
xp = [a_p; b_p; qs_p; ms_p; g_p; spinf_p; sw_p;
    epinfma_p; ewma_p;    
    invefl_p; ccfl_p; yfl_p;
    invel_p; cl_p; pinfl_p; wl_p; yl_p; rl_p;
    kpf_p; kp_p]
yp = [rkf_p; wf_p; zcapf_p; labf_p; kkf_p; invef_p; pkf_p; rrf_p; yf_p; ccf_p;
    mc_p; rk_p; w_p; zcap_p; lab_p; kk_p; inve_p; pk_p; r_p; pinf_p; y_p; c_p]

nx  = length(x)
ny  = length(y)
nvar   = nx + ny

variables = [x; y; xp; yp]

# Shock
@vars ea eb eqs ems eg espinf esw
e = [ea; eb; eqs; ems; eg; espinf; esw]
eta =   Array([ sda 0.0 0.0 0.0 0.0 0.0 0.0; #a
                0.0 sdb 0.0 0.0 0.0 0.0 0.0; #b
                0.0 0.0 sdq 0.0 0.0 0.0 0.0; #qs
                0.0 0.0 0.0 sdm 0.0 0.0 0.0; #ms
                cgy 0.0 0.0 0.0 sdg 0.0 0.0; #g
                0.0 0.0 0.0 0.0 0.0 sdp 0.0; #spinf
                0.0 0.0 0.0 0.0 0.0 0.0 sdw; #sw
                0.0 0.0 0.0 0.0 0.0 sdp 0.0; #pinfma
                0.0 0.0 0.0 0.0 0.0 0.0 sdw; #wma
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #kpf
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #invefl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #ccfl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #yfl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #kp
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #invel
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #cl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #pinfl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #wl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0; #yl
                0.0 0.0 0.0 0.0 0.0 0.0 0.0 ]#rl
                )
ne = length(e)

# Equilibrium conditions
# Exogenous processes
e1 = a_p - crhoa*a # TFP shock
e2 = b_p - crhob*b # Preference shock
e3 = qs_p - crhoqs*qs # Investment shock
e4 = ms_p - crhoms*ms # Monetary shock        
e5 = g_p - crhog*g # Exogenous spending shock
e6 = spinf_p - crhopinf*spinf + cmap*epinfma # Price shock
e7 = sw_p - crhow*sw + cmaw*ewma # Wage shock
e8 = epinfma_p
e9 = ewma_p 

# Lags
l1 = invef - invefl_p
l2 = ccf - ccfl_p
l3 = inve - invel_p
l4 = yf - yfl_p
l5 = c - cl_p
l6 = pinf - pinfl_p
l7 = w - wl_p
l8 = yy - yl_p
l9 = r - rl_p

# Update of endogenous states
u1 = (1.0-cikbar)*kpf + (cikbar)*invef + (cikbar)*(cgamma^2.0*csadjcost)*qs - kpf_p
u2 = (1.0-cikbar)*kp + cikbar*inve + cikbar*cgamma^2.0*csadjcost*qs - kp_p

# Flexible economy
f1 = calfa * rkf + (1.0-calfa)*wf - 0.0*(1.0-calfa)*a - 1.0*a
f2 = (1.0/(czcap/(1.0-czcap)))*rkf - zcapf
f3 = wf + labf - kkf - rkf
f4 = kpf + zcapf - kkf
f5 = (1.0/(1.0 + cbetabar * cgamma))*(invefl + cbetabar*cgamma*invef_p + (1.0/(cgamma^2.0*csadjcost)) * pkf) + qs - invef
f6 = -rrf - 0.0*b + (1.0/((1.0-chabb/cgamma)/(csigma*(1+chabb/cgamma))))*b + (crk/(crk+(1-ctou)))*rkf_p + ((1.0-ctou)/(crk+(1.0-ctou)))*pkf_p - pkf
f7 = (chabb/cgamma)/(1.0+chabb/cgamma)*ccfl + (1.0/(1.0+chabb/cgamma))*ccf_p + ((csigma-1.0)*cwhlc/(csigma*(1+chabb/cgamma)))*(labf - labf_p) - (1.0-chabb/cgamma)/(csigma*(1+chabb/cgamma))*(rrf + 0.0*b) + b - ccf
f8 = ccy*ccf + ciy*invef + g + crkky*zcapf - yf
f9 = cfc*(calfa*kkf + (1.0-calfa)*labf + a) - yf
f10 = csigl*labf + (1.0/(1.0-chabb/cgamma))*ccf - (chabb/cgamma)/(1.0-chabb/cgamma)*ccfl - wf

# Sticky price - wage economy
s1 = calfa * rk + (1.0 - calfa)*(w) - 1.0*a - 0.0*(1.0-calfa)*a - mc 
s2 = (1.0/(czcap/(1.0-czcap)))*rk - zcap
s3 = w + lab - kk - rk
s4 = kp + zcap - kk
s5 = (1.0/(1.0+cbetabar*cgamma))*(invel + cbetabar*cgamma*inve_p + (1.0/(cgamma^2.0*csadjcost))*pk) + qs - inve
s6 = -r + pinf_p - 0.0*b + (1.0/((1.0-chabb/cgamma)/(csigma*(1.0+chabb/cgamma))))*b + (crk/(crk+(1.0-ctou)))*rk_p + ((1 - ctou)/(crk+(1.0-ctou)))*pk_p - pk
s7 = (chabb/cgamma)/(1.0+chabb/cgamma)*cl + (1.0/(1.0+chabb/cgamma))*c_p + ((csigma-1.0)*cwhlc/(csigma*(1.0+chabb/cgamma)))*(lab - lab_p) - (1.0-chabb/cgamma)/(csigma*(1.0+chabb/cgamma))*(r - pinf_p + 0.0*b) + b - c
s8 = ccy*c + ciy*inve + g + 1.0*crkky*zcap - yy
s9 = cfc*(calfa*kk + (1.0-calfa)*lab + a) - yy
s10 = (1.0/(1.0+cbetabar*cgamma*cindp))*(cbetabar*cgamma*pinf_p + cindp*pinfl + ((1.0-cprobp)*(1.0-cbetabar*cgamma*cprobp)/cprobp)/((cfc - 1.0)*curvp+1.0)*mc) + spinf - pinf
s11 = (1.0/(1.0 + cbetabar*cgamma))*wl + (cbetabar*cgamma/(1.0+cbetabar*cgamma))*w_p + (cindw/(1.0+cbetabar*cgamma))*pinfl - (1.0+cbetabar*cgamma*cindw)/(1.0+cbetabar*cgamma)*pinf + (cbetabar*cgamma)/(1.0+cbetabar*cgamma)*pinf_p + (1.0-cprobw)*(1.0-cbetabar*cgamma*cprobw)/((1.0+cbetabar*cgamma)*cprobw)*(1.0/((clandaw-1.0)*curvw+1.0))*(csigl*lab + (1.0/(1.0-chabb/cgamma))*c - ((chabb/cgamma)/(1.0-chabb/cgamma))*cl - w) + 1.0*sw - w

# Monetary policy rule
m1 = crpi*(1.0 - crr)*pinf + cry*(1.0-crr)*(yy - yf) + crdy*(yy - yf - yl + yfl) + crr*rl + ms - r
    
f = [e1; e2; e3; e4; e5; e6; e7; e8; e9;
    l1; l2; l3; l4; l5; l6; l7; l8; l9;
    u1; u2;
    f1; f2; f3; f4; f5; f6; f7; f8; f9; f10;
    s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11;
    m1]
nf = length(f)

PAR_SS  =   [ctou;         # depreciation rate
            clandaw;      # SS markup labor market
            cg;           # exogenous spending GDP-ratio
            curvp;        # curvature Kimball aggregator goods market
            curvw;        # curvature Kimball aggregator labor market
            ctrend / 100 + 1;
            100 / (constebeta + 100); # discount factor
            constepinf / 100 + 1;
            ctrend;       # quarterly trend growth rate to GDP
            constebeta;
            constepinf;   # quarterly SS inflation rate
            constelab;
            calfa;        # labor share in production
            csigma;       # intertemporal elasticity of substitution
            cfc; 
            cgy;
            csadjcost;                   # investment adjustment cost
            chabb;                   # habit persistence 
            cprobw;                   # calvo parameter labor market
            csigl; 
            cprobp;                   # calvo parameter goods market
            cindw;                   # indexation labor market
            cindp;                   # indexation goods market
            czcap;                   # capital utilization
            crpi;                   # Taylor rule reaction to inflation
            crr;                   # Taylor rule interest rate smoothing
            cry;                   # Taylor rule long run reaction to output gap
            crdy;                   # Taylor rule short run reaction to output gap
            crhoa;
            crhob;
            crhog;
            crhoqs;
            crhoms; 
            crhopinf;
            crhow;
            cmap;
            cmaw;
            cfc;
            cbeta * cgamma^(-csigma);
            cpie / (cbeta * cgamma^(-csigma));
            (cbeta^(-1.0)) * (cgamma^csigma) - (1.0 - ctou);
            (calfa^calfa * (1.0 - calfa)^(1.0 - calfa) / (clandap * crk^calfa))^(1.0 / (1.0 - calfa));
            (1.0 - (1.0 - ctou) / cgamma);
            (1.0 - (1.0 - ctou) / cgamma) * cgamma;
            ((1.0 - calfa) / calfa) * (crk / cw);
            cfc * (clk)^(calfa - 1.0);
            cik * cky;
            1.0 - cg - cik * cky;
            crk * cky;
            (1.0 / clandaw) * (1.0 - calfa) / calfa * crk * cky / ccy;
            1.0 - crk * cky;
            sda;
            sdb;
            sdg;
            sdq;
            sdm;
            sdp;
            sdw
            ]
SS = Array{Sym}([])
# PAR_SS = Array{Sym}([])

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

process_model(model; flag_deviation)                

# Parametrization
ctou        = 0.025;                     # depreciation rate
clandaw     = 1.5;                      # SS markup labor market
cg          = 0.18;                     # exogenous spending GDP-ratio
curvp       = 10;                       # curvature Kimball aggregator goods market
curvw       = 10;                       # curvature Kimball aggregator labor market
ctrend      = 0.4312;                   # quarterly trend growth rate to GDP
cgamma      = ctrend / 100 + 1;
constebeta  = 0.1657;
cbeta       = 100 / (constebeta + 100); # discount factor
constepinf  = 0.7869;                   # quarterly SS inflation rate
cpie        = constepinf / 100 + 1;
constelab   = 0.5509;
calfa       = 0.1901;                   # labor share in production
csigma      = 1.3808;                   # intertemporal elasticity of substitution
cfc         = 1.6064; 
cgy         = 0.5187;
csadjcost   = 5.7606;                   # investment adjustment cost
chabb       = 0.7133;                   # habit persistence 
cprobw      = 0.7061;                   # calvo parameter labor market
csigl       = 1.8383; 
cprobp      = 0.6523;                   # calvo parameter goods market
cindw       = 0.5845;                   # indexation labor market
cindp       = 0.2432;                   # indexation goods market
czcap       = 0.5462;                   # capital utilization
crpi        = 2.0443;                   # Taylor rule reaction to inflation
crr         = 0.8103;                   # Taylor rule interest rate smoothing
cry         = 0.0882;                   # Taylor rule long run reaction to output gap
crdy        = 0.2247;                   # Taylor rule short run reaction to output gap
crhoa       = 0.9577;
crhob       = 0.2194;
crhog       = 0.9767;
crhoqs      = 0.7113;
crhoms      = 0.1479; 
crhopinf    = 0.8895;
crhow       = 0.9688;
cmap        = 0.7010;
cmaw        = 0.8503;

clandap     = cfc;
cbetabar    = cbeta * cgamma^(-csigma);
cr          = cpie / (cbeta * cgamma^(-csigma));
crk         = (cbeta^(-1.0)) * (cgamma^csigma) - (1.0 - ctou);
cw          = (calfa^calfa * (1.0 - calfa)^(1.0 - calfa) / (clandap * crk^calfa))^(1.0 / (1.0 - calfa));
cikbar      = (1.0 - (1.0 - ctou) / cgamma);
cik         = (1.0 - (1.0 - ctou) / cgamma) * cgamma;
clk         = ((1.0 - calfa) / calfa) * (crk / cw);
cky         = cfc * (clk)^(calfa - 1.0);
ciy         = cik * cky;
ccy         = 1.0 - cg - cik * cky;
crkky       = crk * cky;
cwhlc       = (1.0 / clandaw) * (1.0 - calfa) / calfa * crk * cky / ccy;
cwly        = 1.0 - crk * cky;

sda         = 0.4618;
sdb         = 1.8513;
sdg         = 0.6090;
sdq         = 0.6017;
sdm         = 0.2397;
sdp         = 0.1455;
sdw         = 0.2089;

PAR = [ctou; clandaw; cg; curvp; curvw;
        cgamma; cbeta; cpie; 
        ctrend; constebeta; constepinf; constelab;
        calfa; csigma; cfc; cgy; 
        csadjcost; chabb; cprobw; csigl; cprobp; cindw; cindp; czcap; crpi; crr; cry; crdy; 
        crhoa; crhob; crhog; crhoqs; crhoms; crhopinf; crhow; cmap; cmaw;
        clandap; cbetabar; cr; crk; cw; cikbar; cik; clk; cky; ciy; ccy; crkky; cwhlc; cwly;
        sda; sdb; sdq; sdm; sdg; sdp; sdw]

eta     =   eval_ShockVAR(PAR)
PAR_SS  =   eval_PAR_SS(PAR)
# SS      =   eval_SS(PAR_SS)
SS      =   SS_solver(model, PAR)
deriv   =   eval_deriv(PAR_SS, SS)

# @btime sol_mat = solve_model(model, deriv, eta)
sol_mat = solve_model(model, deriv, eta)

# ir_logdev = impulseresponse_model(model, sol_mat, 20)
# plot(ir_logdev[:,:])

nsim = 200
# nsim = 500
simulation_logdev = simulate_model(model, sol_mat, nsim)
data = simulation_logdev[10001:10000+nsim,model.nx+1:model.nx+3]

c       = 0.1
npart   = 500 #2^13          # of particles
# npart   = 2^13          # of particles
nphi    = 100 # 500         # of stage
lam     = 3#2.1             # bending coeff
# @time estimation_results = smc_rwmh_threads(model, data, PAR, PAR_SS, eta, SS, deriv, sol_mat, c, npart, nphi, lam)
@time estimation_results = smc_rwmh(model, data, PAR, PAR_SS, eta, SS, deriv, sol_mat, c, npart, nphi, lam)


##################################################################################################################################################################
