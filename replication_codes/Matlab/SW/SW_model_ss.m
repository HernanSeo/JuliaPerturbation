%NEOCLASSICAL_MODEL_SS.M
function [ctou,clandaw,cg,curvp,curvw,cgamma,cbeta,cpie,ctrend,constebeta,...
          constepinf,constelab,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,...
          csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,...
          crhoa,crhob,crhog,crhoqs,crhoms,crhopinf,crhow,...
          cmap,cmaw,clandap,cbetabar,cr,crk,cw,cikbar,cik,clk,...
          cky,ciy,ccy,crkky,cwhlc,cwly,...
          sda,sdb,sdq,sdm,sdg,sdp,sdw,...
          eta,...
          a,b,qs,ms,g,spinf,sw,epinfma,ewma,...
          invefl,cfl,yfl,invel,cl,pinfl,wl,yl,rl,...
          kpf,kp,...
          a_p,b_p,qs_p,ms_p,g_p,spinf_p,sw_p,...
          epinfma_p,ewma_p,...
          invefl_p,cfl_p,yfl_p,...
          invel_p,cl_p,pinfl_p,wl_p,yl_p,rl_p,...
          kpf_p,kp_p,...
          rkf,wf,zcapf,labf,kkf,invef,pkf,rrf,yf,cf,...
          mc,rk,w,zcap,lab,kk,inve,pk,r,pinf,yy,c,...
          rkf_p,wf_p,zcapf_p,labf_p,kkf_p,invef_p,pkf_p,rrf_p,yf_p,cf_p,...
          mc_p,rk_p,w_p,zcap_p,lab_p,kk_p,inve_p,pk_p,r_p,pinf_p,y_p,c_p]...
          =SW_model_ss
%This program produces the the deep structural parameters and computes the steady state of the simple neoclassical growth model described in section 2.1 of ``Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function,'' by Stephanie Schmitt-Grohe and Martin Uribe, (2001). 
%
%(c) Stephanie Schmitt-Grohe and Martin Uribe
%Date July 17, 2001, revised 22-Oct-2004

ctou        = 0.025;                    % depreciation rate
clandaw     = 1.5;                      % SS markup labor market
cg          = 0.18;                     % exogenous spending GDP-ratio
curvp       = 10;                       % curvature Kimball aggregator goods market
curvw       = 10;                       % curvature Kimball aggregator labor market
ctrend      = 0.4312;                   % quarterly trend growth rate to GDP
cgamma      = ctrend / 100 + 1;
constebeta  = 0.1657;
cbeta       = 100 / (constebeta + 100); % discount factor
constepinf  = 0.7869;                   % quarterly SS inflation rate
cpie        = constepinf / 100 + 1;
constelab   = 0.5509;
calfa       = 0.1901;                   % labor share in production
csigma      = 1.3808;                   % intertemporal elasticity of substitution
cfc         = 1.6064; 
cgy         = 0.5187;
csadjcost   = 5.7606;                   % investment adjustment cost
chabb       = 0.7133;                   % habit persistence 
cprobw      = 0.7061;                   % calvo parameter labor market
csigl       = 1.8383; 
cprobp      = 0.6523;                   % calvo parameter goods market
cindw       = 0.5845;                   % indexation labor market
cindp       = 0.2432;                   % indexation goods market
czcap       = 0.5462;                   % capital utilization
crpi        = 2.0443;                   % Taylor rule reaction to inflation
crr         = 0.8103;                   % Taylor rule interest rate smoothing
cry         = 0.0882;                   % Taylor rule long run reaction to output gap
crdy        = 0.2247;                   % Taylor rule short run reaction to output gap
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

a=0;b=0;qs=0;ms=0;g=0;spinf=0;sw=0;epinfma=0;ewma=0;
invefl=0;cfl=0;yfl=0;invel=0;cl=0;pinfl=0;wl=0;yl=0;rl=0;
kpf=0;kp=0;
a_p=0;b_p=0;qs_p=0;ms_p=0;g_p=0;spinf_p=0;sw_p=0;
epinfma_p=0;ewma_p=0;
invefl_p=0;cfl_p=0;yfl_p=0;
invel_p=0;cl_p=0;pinfl_p=0;wl_p=0;yl_p=0;rl_p=0;
kpf_p=0;kp_p=0;
rkf=0;wf=0;zcapf=0;labf=0;kkf=0;invef=0;pkf=0;rrf=0;yf=0;cf=0;
mc=0;rk=0;w=0;zcap=0;lab=0;kk=0;inve=0;pk=0;r=0;pinf=0;yy=0;c=0;
rkf_p=0;wf_p=0;zcapf_p=0;labf_p=0;kkf_p=0;invef_p=0;pkf_p=0;rrf_p=0;yf_p=0;cf_p=0;
mc_p=0;rk_p=0;w_p=0;zcap_p=0;lab_p=0;kk_p=0;inve_p=0;pk_p=0;r_p=0;pinf_p=0;y_p= 0;c_p=0;

eta =[sda 0.0 0.0 0.0 0.0 0.0 0.0; %a
                0.0 sdb 0.0 0.0 0.0 0.0 0.0; %b
                0.0 0.0 sdq 0.0 0.0 0.0 0.0; %qs
                0.0 0.0 0.0 sdm 0.0 0.0 0.0; %ms
                cgy 0.0 0.0 0.0 sdg 0.0 0.0; %g
                0.0 0.0 0.0 0.0 0.0 sdp 0.0; %spinf
                0.0 0.0 0.0 0.0 0.0 0.0 sdw; %sw
                0.0 0.0 0.0 0.0 0.0 sdp 0.0; %pinfma
                0.0 0.0 0.0 0.0 0.0 0.0 sdw; %wma
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
            

