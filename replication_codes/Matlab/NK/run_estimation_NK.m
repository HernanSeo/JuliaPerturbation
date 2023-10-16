clc; clear all;

model.flag_order = 1; model.flag_deviation = true; model.flag_SSsolver = false;

[fx,fxp,fy,fyp,fypyp,fypy,fypxp,fypx,fyyp,fyy,fyxp,fyx,fxpyp,fxpy,fxpxp,fxpx,fxyp,fxy,fxxp,fxx,f] = NK_model;
model.nf = size(f,2);
model.nx = size(fx,2);
model.ny = size(fy,2);

[ALPPHA,BETTA,GAMMA,DELTA,ETA,THETA,XI,RHO,SIG,Z,PHI_PPI,PHI_Y,PHI_R,PPS,YSS,RRS,eta,...
          c,cp,h,hp,l,lp,w,wp,mc,mcp,s,sp,ph,php,ppi,ppip,rt,rtp,x1,x1p,x2,x2p,br,brp,brb,brbp,z,zp,k,kp,ii,iip,u,up]=NK_model_ss;

model.ne = size(eta,2);

model.estimate = ['RHO'; 'SIG'];
model.ns = size(model.estimate,1);
model.priors.RHO.iv = .9676; model.priors.RHO.lb = .01; model.priors.RHO.ub = .9999; model.priors.RHO.d = truncate(makedist('Normal','mu',0.9,'sigma',0.05),0.0,1.0);
model.priors.SIG.iv = .4618; model.priors.SIG.lb = .01; model.priors.SIG.ub = 3.0; model.priors.SIG.d = makedist('Gamma','a',2.0,'b',0.1);

NK_num_eval

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

nsim = 200;
% nsim = 500;
simulation_logdev = simulate_model(model, sol_mat, nsim, eta);
data = simulation_logdev(10001:10000+nsim,model.nx+1:model.nx+3);

c       = 0.1;
npart   = 500;
% npart   = 2^13;          % of particles
nphi    = 100; % 500         % of stage
lam     = 3;%2.1             % bending coeff
tic() 
smc_rwmh(model, data, c, npart, nphi, lam)
estimation_solution = toc()
