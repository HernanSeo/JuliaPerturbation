function smc_rwmh(model, data, c, npart, nphi, lam)
    % Initialization
    phi_bend = true;
    nobs = size(data, 2);
    acpt = 0.25;
    trgt = 0.25;

    % Creating the tempering schedule
    phi_smc = 0:1.0/nphi:1.0;
    if phi_bend
        phi_smc = phi_smc.^lam;
    end

    % Matrices for storing
    wtsim = zeros(npart, nphi);
    zhat = zeros(1, nphi);
    nresamp = 0;

    csim = zeros(1, nphi);
    ESSsim = zeros(1, nphi);
    acptsim = zeros(1, nphi);
    rsmpsim = zeros(1, nphi);
    tic
    % SMC algorithm starts here
    disp("SMC starts ... ")
    parasim = initial_draw(model, npart, nphi);
    wtsim(:, 1) = 1.0/npart * ones(npart, 1);
    zhat(1) = sum(wtsim(:, 1));

    % Posterior values at prior draws
    loglh = zeros(npart, 1);
    logpost = zeros(npart, 1);

    for i = 1:npart
        p0 = reshape(parasim(1, i, :), 1, []);
        loglh(i) = objfun(data, p0, phi_smc, model, nobs);
        prior_val = logpriors(model, p0);
        logpost(i) = loglh(i) + real(prior_val);

        % Handle NaN and Inf values
        if isnan(loglh(i)) || isinf(loglh(i))
            loglh(i) = -1e50;
        end
        if isnan(logpost(i)) || isinf(logpost(i))
            logpost(i) = -1e50;
        end
    end
    toc
    % SMC recursion
    disp("SMC recursion starts ... ")
    estimMean = zeros(nphi, model.ns);

    for i = 2:nphi
        tic
        loglh = real(loglh);

        % Correction
        incwt = exp((phi_smc(i) - phi_smc(i-1)) * loglh);
        wtsim(:, i) = wtsim(:, i-1) .* incwt;
        zhat(i) = sum(wtsim(:, i));
        wtsim(:, i) = wtsim(:, i) / zhat(i);

        % Selection
        ESS = 1.0 / sum(wtsim(:, i).^2.0);

        if ESS < npart/2
            id = randsample(1:npart, npart, true, wtsim(:, i));
            parasim(i-1, :, :) = parasim(i-1, id, :);
            loglh = loglh(id);
            logpost = logpost(id);
            wtsim(:, i) = 1.0 / npart * ones(npart, 1);
            nresamp = nresamp + 1;
            rsmpsim(i) = 1;
        end

        % Mutation
        c = c * (0.95 + 0.10 * exp(16.0 * (acpt - trgt)) / (1 + exp(16.0 * (acpt - trgt))));

        % Calculate estimates of mean & variance
        para = reshape(parasim(i-1, :, :), npart, []);
        wght = wtsim(:, i) * ones(1, model.ns);
        mu = sum(para .* wght, 1);
        z = para - mu .* ones(npart, model.ns);
        R = z' * (z .* wght);

        % Ensure positive definiteness
%         eps = 1e-3;
%         for t = 1:10
%             [leftR, tauR] = eig(R);
%             VtauR = diag(tauR);
%             VtauR(VtauR < eps) = eps;
%             R = leftR * diag(VtauR) * leftR';
%         end

%         Rdiag = diag(R);
%         Rchol = chol(R);
%         Rchol2 = sqrt(Rdiag);

        estimMean(i, :) = mu;

        % Particle mutation
        temp_acpt = zeros(npart, 1);
        for j = 1:npart
            [ind_para, ind_loglh, ind_post, ind_acpt] = mutation_RWMH(model, data, reshape(para(j, :), 1, []), loglh(j), logpost(j), c, R, model.ns, phi_smc, nobs, i);
            parasim(i, j, :) = ind_para;
            loglh(j) = ind_loglh;
            logpost(j) = ind_post;
            temp_acpt(j) = ind_acpt;
        end
        acpt = mean(temp_acpt);

        % Store
        csim(i) = c;
        ESSsim(i) = ESS;
        acptsim(i) = acpt;

        % Print information
        if mod(i, 1) == 0
            disp(".............");
            disp(['phi = ', num2str(phi_smc(i))]);
            disp(['c = ', num2str(c)]);
            disp(['acpt = ', num2str(acpt)]);
            disp(['ESS = ', num2str(ESS), ', ', num2str(nresamp)]);
            disp(".............");
        end
        toc
    end

    % Report summary statistics
    para = reshape(parasim(nphi, :, :), npart, []);
    wght = repmat(wtsim(:, nphi), 1, size(para, 2));
    mu = sum(para .* wght, 1);
    sig = sum((para - repmat(mu, npart, 1)).^2 .* wght, 1);
    sig = sqrt(sig);

    disp(['mu: ', num2str(mu)]);
    disp(['sig: ', num2str(sig)]);
end