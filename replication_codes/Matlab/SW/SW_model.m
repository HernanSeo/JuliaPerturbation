%NEOCLASSICAL_MODEL.M
function [fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,f] = SW_model
%This program computes analytical first and second derivatives of the  function f for the simple neoclassical growth model described in section 2 of ``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function,'' (Journal of Economic Dynamics and Control, 28, January 2004, p. 755-775) by Stephanie Schmitt-Grohe and Martin Uribe. Unlike the example in section 2, here y and x are defined as log(c) and [log(k); log(A)] respectively.   The function f defines  the DSGE model: 
%  E_t f(yp,y,xp,x) =0. 
%
%Inputs: none
%
%Output: Analytical first and second derivatives of the function f
%
%Calls: anal_deriv.m 
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date July 17, 2001, revised 22-Oct-2004

%Define parameters
syms ctou clandaw cg curvp curvw cgamma cbeta cpie ctrend constebeta
syms constepinf constelab calfa csigma cfc cgy csadjcost chabb cprobw
syms csigl cprobp cindw cindp czcap crpi crr cry crdy
syms crhoa crhob crhog crhoq crhom crhop crhow
syms cmap cmaw clandap cbetabar cr crk cw cikbar cik clk
syms cky ciy ccy crkky cwhlc cwly
syms cstda cstdb cstdq cstdm cstdg cstdp cstdw

%Define variables 
syms a b qs ms g spinf sw
syms epinfma ewma
syms invefl cfl yfl
syms invel cl pinfl wl yl rl 
syms kpf kp
syms a_p b_p qs_p ms_p g_p spinf_p sw_p
syms epinfma_p ewma_p
syms invefl_p cfl_p yfl_p
syms invel_p cl_p pinfl_p wl_p yl_p rl_p
syms kpf_p kp_p
syms rkf wf zcapf labf kkf invef pkf rrf yf cf
syms mc rk w zcap lab kk inve pk r pinf yy c
syms rkf_p wf_p zcapf_p labf_p kkf_p invef_p pkf_p rrf_p yf_p cf_p   
syms mc_p rk_p w_p zcap_p lab_p kk_p inve_p pk_p r_p pinf_p y_p c_p

% Define the vector of controls, y, and states, x
x = [a b qs ms g spinf sw...
    epinfma ewma...
    invefl cfl yfl...
    invel cl pinfl wl yl rl...
    kpf kp];
y = [rkf wf zcapf labf kkf invef pkf rrf yf cf...
    mc rk w zcap lab kk inve pk r pinf yy c];
xp = [a_p b_p qs_p ms_p g_p spinf_p sw_p...
    epinfma_p ewma_p... 
    invefl_p cfl_p yfl_p...
    invel_p cl_p pinfl_p wl_p yl_p rl_p...
    kpf_p kp_p];
yp = [rkf_p wf_p zcapf_p labf_p kkf_p invef_p pkf_p rrf_p yf_p cf_p...
    mc_p rk_p w_p zcap_p lab_p kk_p inve_p pk_p r_p pinf_p y_p c_p];


%Write equations fi% Exogenous processes
% Exogenous shocks
e1 = a_p - crhoa*a; % TFP shock
e2 = b_p - crhob*b; % Preference shock
e3 = qs_p - crhoq*qs; % Investment shock
e4 = ms_p - crhom*ms; % Monetary shock        
e5 = g_p - crhog*g; % Exogenous spending shock
e6 = spinf_p - crhop*spinf + cmap*epinfma; % Price shock
e7 = sw_p - crhow*sw + cmaw*ewma; % Wage shock
e8 = epinfma_p;
e9 = ewma_p; 

% Lags
l1 = invef - invefl_p;
l2 = cf - cfl_p;
l3 = inve - invel_p;
l4 = yf - yfl_p;
l5 = c - cl_p;
l6 = pinf - pinfl_p;
l7 = w - wl_p;
l8 = yy - yl_p;
l9 = r - rl_p;

% Update of endogenous states
u1 = (1.0-cikbar)*kpf + (cikbar)*invef + (cikbar)*(cgamma^2.0*csadjcost)*qs - kpf_p;
u2 = (1.0-cikbar)*kp + cikbar*inve + cikbar*cgamma^2.0*csadjcost*qs - kp_p;

% Flexible economy
f1 = calfa * rkf + (1.0-calfa)*wf - 0.0*(1.0-calfa)*a - 1.0*a;
f2 = (1.0/(czcap/(1.0-czcap)))*rkf - zcapf;
f3 = wf + labf - kkf - rkf;
f4 = kpf + zcapf - kkf;
f5 = (1.0/(1.0 + cbetabar * cgamma))*(invefl + cbetabar*cgamma*invef_p + (1.0/(cgamma^2.0*csadjcost)) * pkf) + qs - invef;
f6 = -rrf - 0.0*b + (1.0/((1.0-chabb/cgamma)/(csigma*(1+chabb/cgamma))))*b + (crk/(crk+(1-ctou)))*rkf_p + ((1.0-ctou)/(crk+(1.0-ctou)))*pkf_p - pkf;
f7 = (chabb/cgamma)/(1.0+chabb/cgamma)*cfl + (1.0/(1.0+chabb/cgamma))*cf_p + ((csigma-1.0)*cwhlc/(csigma*(1+chabb/cgamma)))*(labf - labf_p) - (1.0-chabb/cgamma)/(csigma*(1+chabb/cgamma))*(rrf + 0.0*b) + b - cf;
f8 = ccy*cf + ciy*invef + g + crkky*zcapf - yf;
f9 = cfc*(calfa*kkf + (1.0-calfa)*labf + a) - yf;
f10 = csigl*labf + (1.0/(1.0-chabb/cgamma))*cf - (chabb/cgamma)/(1.0-chabb/cgamma)*cfl - wf;

% Sticky price - wage economy
s1 = calfa * rk + (1.0 - calfa)*(w) - 1.0*a - 0.0*(1.0-calfa)*a - mc; 
s2 = (1.0/(czcap/(1.0-czcap)))*rk - zcap;
s3 = w + lab - kk - rk;
s4 = kp + zcap - kk;
s5 = (1.0/(1.0+cbetabar*cgamma))*(invel + cbetabar*cgamma*inve_p + (1.0/(cgamma^2.0*csadjcost))*pk) + qs - inve;
s6 = -r + pinf_p - 0.0*b + (1.0/((1.0-chabb/cgamma)/(csigma*(1.0+chabb/cgamma))))*b + (crk/(crk+(1.0-ctou)))*rk_p + ((1 - ctou)/(crk+(1.0-ctou)))*pk_p - pk;
s7 = (chabb/cgamma)/(1.0+chabb/cgamma)*cl + (1.0/(1.0+chabb/cgamma))*c_p + ((csigma-1.0)*cwhlc/(csigma*(1.0+chabb/cgamma)))*(lab - lab_p) - (1.0-chabb/cgamma)/(csigma*(1.0+chabb/cgamma))*(r - pinf_p + 0.0*b) + b - c;
s8 = ccy*c + ciy*inve + g + 1.0*crkky*zcap - yy;
s9 = cfc*(calfa*kk + (1.0-calfa)*lab + a) - yy;
s10 = (1.0/(1.0+cbetabar*cgamma*cindp))*(cbetabar*cgamma*pinf_p + cindp*pinfl + ((1.0-cprobp)*(1.0-cbetabar*cgamma*cprobp)/cprobp)/((cfc - 1.0)*curvp+1.0)*mc) + spinf - pinf;
s11 = (1.0/(1.0 + cbetabar*cgamma))*wl + (cbetabar*cgamma/(1.0+cbetabar*cgamma))*w_p + (cindw/(1.0+cbetabar*cgamma))*pinfl - (1.0+cbetabar*cgamma*cindw)/(1.0+cbetabar*cgamma)*pinf + (cbetabar*cgamma)/(1.0+cbetabar*cgamma)*pinf_p + (1.0-cprobw)*(1.0-cbetabar*cgamma*cprobw)/((1.0+cbetabar*cgamma)*cprobw)*(1.0/((clandaw-1.0)*curvw+1.0))*(csigl*lab + (1.0/(1.0-chabb/cgamma))*c - ((chabb/cgamma)/(1.0-chabb/cgamma))*cl - w) + 1.0*sw - w;

% Monetary policy rule
m1 = crpi*(1.0 - crr)*pinf + cry*(1.0-crr)*(yy - yf) + crdy*(yy - yf - yl + yfl) + crr*rl + ms - r;
    
f = [e1; e2; e3; e4; e5; e6; e7; e8; e9;...
    l1; l2; l3; l4; l5; l6; l7; l8; l9;...
    u1; u2;...
    f1; f2; f3; f4; f5; f6; f7; f8; f9; f10;...
    s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11;...
    m1];

%Make f a function of the logarithm of the state and control vector
% f = subs(f, [x,y,xp,yp], (exp([x,y,xp,yp])));
%if line 36 gives an error (whichh it will for some versions of Matlab), percentage line 36 out and instead use line 38.
%f = subs(f, [x,y,xp,yp], transpose(exp([x,y,xp,yp])));

ETASHOCK =[cstda 0.0 0.0 0.0 0.0 0.0 0.0; %a
            0.0 cstdb 0.0 0.0 0.0 0.0 0.0; %b
            0.0 0.0 cstdq 0.0 0.0 0.0 0.0; %qs
            0.0 0.0 0.0 cstdm 0.0 0.0 0.0; %ms
            cgy 0.0 0.0 0.0 cstdg 0.0 0.0; %g
            0.0 0.0 0.0 0.0 0.0 cstdp 0.0; %spinf
            0.0 0.0 0.0 0.0 0.0 0.0 cstdw; %sw
            0.0 0.0 0.0 0.0 0.0 cstdp 0.0; %pinfma
            0.0 0.0 0.0 0.0 0.0 0.0 cstdw; %wma
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %kpf
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %invefl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %cfl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %yfl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %kp
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %invel
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %cl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %pinfl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %wl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0; %yl
            0.0 0.0 0.0 0.0 0.0 0.0 0.0];%rl



%Compute analytical derivatives of f
[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx]=anal_deriv(f,x,y,xp,yp);
anal_deriv_print2f('SW',fx,fxp,fy,fyp,f,ETASHOCK, fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx)