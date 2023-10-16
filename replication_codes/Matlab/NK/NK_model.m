%NEOCLASSICAL_MODEL.M
function [fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,f] = NK_model
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
syms ALPPHA BETTA GAMMA DELTA ETA THETA XI RHO SIG PHI_R PHI_PPI PHI_Y PPS YSS RRS

%Define variables 
syms c cp h hp l lp w wp mc mcp s sp ph php ppi ppip rt rtp x1 x1p x2 x2p br brp brb brbp z zp k kp ii iip u up

%Write equations fi, i=1:3
f1 = -kp + (1-DELTA)*k + ii;
f2  = l - (c*(1-h)^GAMMA)^(-XI) * (1-h)^GAMMA;
f3  = BETTA*lp/ppip - l*rt;
f4  = 1/br - rt;
f5  = -sp + (1-ALPPHA)*(ph^(-ETA)) + ALPPHA*(ppi^ETA)*s;
f6  = z*h^(1-THETA)*k^(THETA) - sp*(c+ii);
f7  = -1 + (1-ALPPHA)*(ph^(1-ETA)) + ALPPHA*(ppi^(ETA-1));
f8  = -x1 + (ph^(1-ETA))*c*((ETA-1)/ETA) + ALPPHA*rt*((ph/php)^(1-ETA))*(ppip^(ETA))*x1p;
f9  = -x2 + (ph^(-ETA))*c*mc + ALPPHA*rt*((ph/php)^(-ETA))*(ppip^(ETA+1))*x2p;
f10 = x2-x1;
f11 = w - mc*z*(1-THETA)*h^(-THETA)*k^(THETA);
f12 = w - ((c*(1-h)^GAMMA)^(-XI)*GAMMA*(1-h)^(GAMMA-1)*c)/((c*(1-h)^GAMMA)^(-XI)*(1-h)^GAMMA);
f13a = u - mc*z*THETA*h^(1-THETA)*k^(THETA-1);
f13b = l - BETTA*(up+1-DELTA)*lp;
f14a = log(br/RRS) - PHI_R*log(brb/RRS) - PHI_PPI*log(ppi/PPS) - PHI_Y*log((sp*(c+ii))/YSS);
f14b = brbp - br;
f15 = log(zp) - RHO * log(z);

%Create function f
f = [f1;f2;f3;f4;f5;f6;f7;f8;f9;f10;f11;f12;f13a;f13b;f14a;f14b;f15];

% Define the vector of controls, y, and states, x
x = [s z brb k];
y = [c h ppi rt br w l mc ph x1 x2 ii u];
xp = [sp zp brbp kp];
yp = [cp hp ppip rtp brp wp lp mcp php x1p x2p iip up];

%Make f a function of the logarithm of the state and control vector
f = subs(f, [x,y,xp,yp], (exp([x,y,xp,yp])));
%if line 36 gives an error (whichh it will for some versions of Matlab), percentage line 36 out and instead use line 38.
%f = subs(f, [x,y,xp,yp], transpose(exp([x,y,xp,yp])));

ETASHOCK =[0 SIG 0 0];
            
%Compute analytical derivatives of f
[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx]=anal_deriv(f,x,y,xp,yp);
anal_deriv_print2f('NK',fx,fxp,fy,fyp,f,ETASHOCK, fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx);