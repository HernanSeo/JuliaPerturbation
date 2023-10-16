clc; clear all;

model.flag_order = 1; model.flag_deviation = false; model.flag_SSsolver = false;

[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,f] = SW_model;

[ctou,clandaw,cg,curvp,curvw,cgamma,cbeta,cpie,ctrend,constebeta,...
      constepinf,constelab,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,...
      csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,...
      crhoa,crhob,crhog,crhoq,crhom,crhop,crhow,...
      cmap,cmaw,clandap,cbetabar,cr,crk,cw,cikbar,cik,clk,...
      cky,ciy,ccy,crkky,cwhlc,cwly,...
      cstda,cstdb,cstdq,cstdm,cstdg,cstdp,cstdw,...
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
      mc_p,rk_p,w_p,zcap_p,lab_p,kk_p,inve_p,pk_p,r_p,pinf_p,y_p,c_p] =...
      SW_model_ss;

%Obtain numerical derivatives of f
SW_num_eval

tic()
for t=1:100    
    %First-order approximation
    [sol_mat.gx,sol_mat.hx] = gx_hx(nfy,nfx,nfyp,nfxp);

    if model.flag_order > 1
        %Second-order approximation
        [sol_mat.gxx,sol_mat.hxx] = gxx_hxx(nfx,nfxp,nfy,nfyp,nfypyp,nfypy,nfypxp,nfypx,nfyyp,nfyy,nfyxp,nfyx,nfxpyp,nfxpy,nfxpxp,nfxpx,nfxyp,nfxy,nfxxp,nfxx,sol_mat.hx,sol_mat.gx); 
        [sol_mat.gss,sol_mat.hss] = gss_hss(nfx,nfxp,nfy,nfyp,nfypyp,nfypy,nfypxp,nfypx,nfyyp,nfyy,nfyxp,nfyx,nfxpyp,nfxpy,nfxpxp,nfxpx,nfxyp,nfxy,nfxxp,nfxx,sol_mat.hx,sol_mat.gx,sol_mat.gxx,eta);
    end
end    
time_solution = toc();
time_solution = time_solution*10^4 

model.estimate = ['crhoa'; 'cstda'];
priors.crhoa.iv = .9676; priors.crhoa.lb = .01; priors.crhoa.ub = .9999; priors.crhoa.d = truncate(makedist('Normal','mu',0.9,'sigma',0.05),0.0,1.0);
priors.cstda.iv = 0.4618; priors.cstda.lb = 0.01; priors.cstda.ub = 3.0; priors.cstda.d = makedist('Gamma','a',2.0,'b',0.1);

model.ne = size(eta,2);
model.ns = size(model.estimate,1);
model.nx = size(fx,2);
model.ny = size(fy,2);
model.priors = priors;

nsim = 200;
% nsim = 500;
simulation_logdev = simulate_model(model, sol_mat, nsim, eta);
data = simulation_logdev(10001:10000+nsim,model.nx+1:model.nx+3);

c       = 0.1;
npart   = 500;%2^13          % of particles
% npart   = 2^13;          % of particles
nphi    = 100; % 500         % of stage
lam     = 3;%2.1             % bending coeff
tic() 
smc_rwmh(model, data, c, npart, nphi, lam)
time_estimation = toc()