function [ind_para, ind_loglh, ind_post, ind_acpt] = mutation_RWMH(model, data, p0, l0, lpost0, c, R, npara, phi_smc, nobs, i)
    % RW proposal
    px = p0 + ((c * chol(R, 'upper')' * randn(npara,1)))';
    
    prior_val = logpriors(model, px);
    if prior_val == -Inf
        lnpost = -Inf;
        lnpY = -Inf;
        lnprio = -Inf;
        error = 10;
    else
        lnpY = objfun(data, px, phi_smc, model, nobs);
        lnpost = lnpY + prior_val;
    end
    
    % Accept/Reject
    alp = exp(lnpost - lpost0); % this is RW, so q is canceled out

    if rand() < alp % accept
        ind_para = px;
        ind_loglh = lnpY;
        ind_post = lnpost;
        ind_acpt = 1;
    else
        ind_para = p0;
        ind_loglh = l0;
        ind_post = lpost0;
        ind_acpt = 0;
    end
end
