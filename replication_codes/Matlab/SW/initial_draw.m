function parasim = initial_draw(model, npart, nphi)
    ns = model.ns;
    parasim = zeros(nphi, npart, ns);
    fn=fieldnames(model.priors);
    
    for i = 1:size(fn)
        parasim(1, :, i) = random(model.priors.(fn{i}).d, [1, npart]);
    end
    % You can add similar code for other parameters if needed.
end