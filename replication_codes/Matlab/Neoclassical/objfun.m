function result = objfun(data, p0, phi_smc, model, nobs)
    t0 = 1;
    [SIG,DELTA,ALFA,BETTA,RHO,MUU,AA,eta,cc,cp,nn,np,yy,yp,rr,rp,ii,ip,k,kp,a,ap,A,K,C,R,Y,I]=neoclassical_model_ss;

    for i = 1:model.ns
        sfg = strcat(genvarname(model.estimate(i,:)),' = p0(',string(i),');');
        eval(sfg); 
    end

    neoclassical_num_eval;
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