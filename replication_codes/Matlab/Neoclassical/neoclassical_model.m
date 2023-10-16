%NEOCLASSICAL_MODEL.M
function [fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,f] = neoclassical_model
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
syms 	SIG DELTA ALFA BETTA RHO MUU AA

%Define variables 
syms cc cp yy yp rr rp nn np ii ip k kp a ap 

%Write equations fi, i=1:3
f1 = cc + kp - (1-DELTA) * k - a * k^ALFA * nn^(1-ALFA);
f2 = cc^(-SIG) - BETTA * cp^(-SIG) * (ap * ALFA * kp^(ALFA-1) * np^(1-ALFA) + 1 - DELTA);
f3 = AA - cc^(-SIG) * a * (1-ALFA) * k^ALFA * nn^(-ALFA); 
f4 = log(ap) - RHO * log(a);
f5 = rr - (a * ALFA * k^(ALFA-1) * nn^(1-ALFA));
f6 = yy - a * k^ALFA * nn^(1-ALFA);
f7 = ii - (kp - (1-DELTA) * k);
% f1 = cc + kp - (1-DELTA) * k - a * k^ALFA;
% f2 = cc^(-SIG) - BETTA * cp^(-SIG) * (ap * ALFA * kp^(ALFA-1));
% f3 = log(ap) - RHO * log(a);

%Create function f
f = [f1;f2;f3;f4;f5;f6;f7];
% f = [f1;f2;f3];

% Define the vector of controls, y, and states, x
x = [k a];
y = [cc rr nn yy ii];
% y = [cc];
xp = [kp ap];
yp = [cp rp np yp ip];
% yp = [cp];

%Make f a function of the logarithm of the state and control vector
f = subs(f, [x,y,xp,yp], (exp([x,y,xp,yp])));
%if line 36 gives an error (whichh it will for some versions of Matlab), percentage line 36 out and instead use line 38.
%f = subs(f, [x,y,xp,yp], transpose(exp([x,y,xp,yp])));

ETASHOCK = [0 MUU]; 
    
%Matrix defining driving force
%Compute analytical derivatives of f
[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx]=anal_deriv(f,x,y,xp,yp);
anal_deriv_print2f('neoclassical',fx,fxp,fy,fyp,f,ETASHOCK, fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx);