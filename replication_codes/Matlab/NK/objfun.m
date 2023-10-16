function result = objfun(data, p0, phi_smc, model, nobs)
        t0 = 1;
        [ALPPHA,BETTA,GAMMA,DELTA,ETA,THETA,XI,RHO,SIG,Z,PHI_PPI,PHI_Y,PHI_R,PPS,YSS,RRS,eta,...
          c,cp,h,hp,l,lp,w,wp,mc,mcp,s,sp,ph,php,ppi,ppip,rt,rtp,x1,x1p,x2,x2p,br,brp,brb,brbp,z,zp,k,kp,ii,iip,u,up] =...
          NK_model_ss;
      
    for i = 1:model.ns
        sfg = strcat(genvarname(model.estimate(i,:)),' = p0(',string(i),');');
        eval(sfg) 
    end
    
    NK_num_eval;
    eta = nETASHOCK';
    
    %First-order approximation
    [sol_mat.gx,sol_mat.hx, sol_mat.qzflag] = gx_hx(nfy,nfx,nfyp,nfxp);
    
    if model.flag_order > 1
        %Second-order approximation
        [sol_mat.gxx,sol_mat.hxx] = gxx_hxx(nfx,nfxp,nfy,nfyp,nfypyp,nfypy,nfypxp,nfypx,nfyyp,nfyy,nfyxp,nfyx,nfxpyp,nfxpy,nfxpxp,nfxpx,nfxyp,nfxy,nfxxp,nfxx,sol_mat.hx,sol_mat.gx); 
        [sol_mat.gss,sol_mat.hss] = gss_hss(nfx,nfxp,nfy,nfyp,nfypyp,nfypy,nfypxp,nfypx,nfyyp,nfyy,nfyxp,nfyx,nfxpyp,nfxpy,nfxpxp,nfxpx,nfxyp,nfxy,nfxxp,nfxx,sol_mat.hx,sol_mat.gx,sol_mat.gxx,eta);
    end

    if sol_mat.qzflag == 1
        result = kf(data, sol_mat.hx, eta*eta', zeros(nobs, 1), eye(nobs, model.ny) * sol_mat.gx, 0.0001 * eye(nobs), t0);
    else
        result = -10000000000.0;
    end
end