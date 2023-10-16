function [fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx]=anal_deriv(f,x,y,xp,yp,approx); 
%[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx]=anal_deriv(f,x,y,xp,yp,approx); 
% Computes analytical first and second (if approx=2) derivatives of the function f(yp,y,xp,x) with respect to x, y, xp, and yp.  For documentation, see the paper ``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function,'' by Stephanie Schmitt-Grohe and Martin Uribe, 2001). 
%
%Inputs: f, x, y, xp, yp, approx
%
%Output: Analytical first and second derivatives of f. 
%
%If approx is set at a value different from 2, the program delivers the first derivatives of f and sets second derivatives at zero. If approx equals 2, the program returns first and second derivatives of f. The default value of approx is 2. 
%Note: This program requires MATLAB's Symbolic Math Toolbox
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date July 17, 2001


if nargin==5
approx=2;
end

nx = size(x,2);
ny = size(y,2);
nxp = size(xp,2);
nyp = size(yp,2);

n = size(f,1);

%Compute the first and second derivatives of f
fx = jacobian(f,x);
fxp = jacobian(f,xp);
fy = jacobian(f,y);
fyp = jacobian(f,yp);

if approx==2

fypyp = reshape(jacobian(fyp(:),yp),n,nyp,nyp);

fypy = reshape(jacobian(fyp(:),y),n,nyp,ny);

fypxp = reshape(jacobian(fyp(:),xp),n,nyp,nxp);

fypx = reshape(jacobian(fyp(:),x),n,nyp,nx);

fyyp = reshape(jacobian(fy(:),yp),n,ny,nyp);

fyy = reshape(jacobian(fy(:),y),n,ny,ny);

fyxp = reshape(jacobian(fy(:),xp),n,ny,nxp);

fyx = reshape(jacobian(fy(:),x),n,ny,nx);

fxpyp = reshape(jacobian(fxp(:),yp),n,nxp,nyp);

fxpy = reshape(jacobian(fxp(:),y),n,nxp,ny);

fxpxp = reshape(jacobian(fxp(:),xp),n,nxp,nxp);

fxpx = reshape(jacobian(fxp(:),x),n,nxp,nx);

fxyp = reshape(jacobian(fx(:),yp),n,nx,nyp);

fxy = reshape(jacobian(fx(:),y),n,nx,ny);

fxxp = reshape(jacobian(fx(:),xp),n,nx,nxp);

fxx = reshape(jacobian(fx(:),x),n,nx,nx);

else

fypyp=0; fypy=0; fypxp=0; fypx=0; fyyp=0; fyy=0; fyxp=0; fyx=0; fxpyp=0; fxpy=0; fxpxp=0; fxpx=0; fxyp=0; fxy=0; fxxp=0; fxx=0;

end 