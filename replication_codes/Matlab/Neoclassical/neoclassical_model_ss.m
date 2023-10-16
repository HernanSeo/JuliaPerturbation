%NEOCLASSICAL_MODEL_SS.M
function [SIG,DELTA,ALFA,BETTA,RHO,MUU,AA,eta,cc,cp,nn,np,yy,yp,rr,rp,ii,ip,k,kp,a,ap,A,K,C,R,Y,I]=neoclassical_model_ss
%This program produces the the deep structural parameters and computes the steady state of the simple neoclassical growth model described in section 2.1 of ``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function,'' by Stephanie Schmitt-Grohe and Martin Uribe, (2001). 
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date July 17, 2001, revised 22-Oct-2004

BETTA=0.95; %discount rate
DELTA=1; %depreciation rate
ALFA=0.3; %capital share
RHO=0.8; %persistence of technology shock
SIG=2; %intertemporal elasticity of substitution
MUU=.05;
eta=[0 MUU]'; %Matrix defining driving force

A   =   1.0;
N   =   2/3;
K   =   (ALFA/(1/BETTA-1+DELTA))^(1/(1-ALFA))*N;
C   =   A * K^(ALFA) * N^(1-ALFA) - DELTA*K;
R   =   A * ALFA * K^(ALFA-1.0) * N^(1-ALFA);
Y  =   A * K^ALFA * N^(1-ALFA);
I  =   DELTA*K;
AA  = C^(-SIG) * (1-ALFA)*A*K^(ALFA)*N^(-ALFA);

a = log(A); 
k = log(K);
cc = log(C);
nn = log(N);
rr = log(R);
yy = log(Y);
ii = log(I);


ap=a;
kp=k;
cp=cc;
np=nn;
yp=yy;
rp=rr;
ip=ii;