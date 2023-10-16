function lnp = logpriors(model, p0)
    nparams = numel(p0);
    idx = NaN * zeros(nparams,1);
    fn=fieldnames(model.priors);
    
    for i = 1:size(fn)
        if p0(i) < model.priors.(fn{i}).lb || p0(i) > model.priors.(fn{i}).ub
            idx(i) = 0.0;
        else
            idx(i) = log(pdf(model.priors.(fn{i}).d, p0(i)));
        end
    end

    lnp = sum(log(idx));
end
