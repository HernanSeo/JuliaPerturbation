%NEOCLASSICAL_MODEL_SS.M
function [ALPPHA,BETTA,GAMMA,DELTA,ETA,THETA,XI,RHO,SIG,Z,PHI_PPI,PHI_Y,PHI_R,PPS,YSS,RRS,eta,...
          c,cp,h,hp,l,lp,w,wp,mc,mcp,s,sp,ph,php,ppi,ppip,rt,rtp,x1,x1p,x2,x2p,br,brp,brb,brbp,z,zp,k,kp,ii,iip,u,up...
          ]=NK_model_ss
%This program produces the the deep structural parameters and computes the steady state of the simple neoclassical growth model described in section 2.1 of ``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function,'' by Stephanie Schmitt-Grohe and Martin Uribe, (2001). 
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date July 17, 2001, revised 22-Oct-2004

ALPPHA = 0.8;
BETTA = 1.04^(-1/4);
GAMMA = 3.6133;
DELTA = 0.01;
ETA = 5;
THETA = 0.3;
XI = 2;
RHO = 0.8556;
Z = 1;
SIG = 0.05;
PHI_PPI = 3;
PHI_Y = 0.01;
PHI_R = 0.8;
PPS = 1.042^(1/4);
RRS   = PPS/BETTA;
eta=[0 SIG 0 0]';

R = 1/RRS;
RR= RRS;
PH=((1-ALPPHA*PPS^(ETA-1))/(1-ALPPHA))^(1/(1-ETA));
PPI=PPS;
S = (1-ALPPHA)*PH^(-ETA)/(1-ALPPHA*PPS^(ETA));
MC = ((1-ALPPHA*RRS^(-1)*PPS^(ETA+1))*((ETA-1)/ETA)*PH)/(1-ALPPHA*RRS^(-1)*PPS^(ETA));
U = 1/BETTA - 1 + DELTA;
H = ((1-THETA)*MC*Z*((THETA*Z*MC)/U)^(THETA/(1-THETA)))/((1-THETA)*MC*Z*((THETA*Z*MC)/U)^(THETA/(1-THETA)) + (GAMMA*Z*((THETA*Z*MC)/U)^(THETA/(1-THETA)) - S*DELTA*((THETA*Z*MC)/U)^(1/(1-THETA)))/S);
K = ((THETA*Z*MC)/U)^(1/(1-THETA))*H;
W = (1-THETA)*MC*Z*(K/H)^THETA;
Y = Z*H^(1-THETA)*K^THETA;
YSS   = Y;
C = (Y - S*DELTA*K)/S;
II = DELTA*K;
X1 = (PH^(1-ETA)*Y*S^(-1)*((ETA-1)/ETA))/(1-ALPPHA*RRS^(-1)*PPS^ETA);
X2 = X1;
LAM = C^(-XI)*(1-H)^(GAMMA*(1-XI));
       
%         YSS = 0.40249214260064275


c = log(C);
cp = c;
h = log(H); 
hp = h;
l = log(LAM);
lp = l;
w = log(W);
wp = w;
mc = log(MC);
mcp = mc;
s = log(S);
sp = s;
ph = log(PH);
php = ph;
ppi = log(PPI);
ppip = ppi;
rt = log(R);
rtp = rt;
x1 = log(X1);
x1p = x1;
x2 = log(X2);
x2p = x2;
br = log(RR);
brp = br;
brb = log(RR);
brbp = brb;
z = log(Z);
zp = z; 
k = log(K);
kp = k;
ii = log(II);
iip = ii;
u = log(U);
up = u;