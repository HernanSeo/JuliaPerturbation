%GSS_HSS.M
%Finds the vectors gss and hss necessary to compute the 2nd order approximation to the decision rules of a DSGE model. For documentation, see the paper ``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function,'' by Stephanie Schmitt-Grohe and Martin Uribe, 2001)
%
%INPUTS: fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,hx,gx,gxx,eta%OUTPUTS: hss gss
%
% We solve a linear system of the type q = Q * x where x=[gss; hss];
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date February 18, 2004


function [gss,hss] = gss_hss(fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,hx,gx,gxx,eta)

nx = size(hx,1); %rows of hx and hss
ny = size(gx,1); %rows of gx and gss
n = nx + ny;
ne = size(eta,2); %number of exogenous shocks (columns of eta)
 
for i=1:n

%First Term
Qh(i,:) = fyp(i,:) * gx;

%Second Term
q(i,1) = sum( diag( ( shiftdim(fypyp(i,:,:),1) * gx * eta)' * gx * eta ));

%Third Term
q(i,1) = q(i,1) + sum( diag(( shiftdim(fypxp(i,:,:),1) *  eta)' * gx * eta ));


%Fourth Term
fyp(i,:) * reshape(gxx,ny,nx^2);

q(i,1) =  q(i,1) + sum( diag(( reshape(ans,nx,nx) * eta )' * eta ));
  
%Fifth Term
Qg(i,:) = fyp(i,:);

%Sixth Term
Qg(i,:) = Qg(i,:) + fy(i,:);

%Seventh Term
Qh(i,:) = Qh(i,:) + fxp(i,:);

%Eighth Term
q(i,1) = q(i,1) + sum( diag( ( shiftdim(fxpyp(i,:,:),1) * gx * eta)' * eta ));

%Nineth Term
q(i,1) = q(i,1) + sum(diag( ( shiftdim(fxpxp(i,:,:),1) * eta)' * eta ));


end %i

x=-([Qg Qh])\q;

gss = x(1:ny);
hss = x(ny+1:end);

