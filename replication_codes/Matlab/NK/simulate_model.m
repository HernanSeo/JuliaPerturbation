function [sim_data] = simulate_model(model, sol_mat, nsim, eta)
    ne = model.ne;
    nx = model.nx;
    ny = model.ny;

    sim_shocks = randn(nsim+10000, ne);
    sim_x = zeros(nsim+10000, nx);
    sim_y = zeros(nsim+10000, ny);

    if model.flag_order > 1
        sim_x_f = zeros(nsim+10000, nx);
        sim_x_s = zeros(nsim+10000, nx);

        for t = 1:nsim+10000
            for i = 1:ny
                sim_y(t,i) = sol_mat.gx(i,:) * (sim_x_f(t,:) + sim_x_s(t,:))' + 1/2 * sim_x_f(t,:) * squeeze(sol_mat.gxx(i,:,:)) * sim_x_f(t,:)' + 1/2 * sol_mat.gss(i);
            end

            if t < nsim+10000
                for i = 1:nx
                    sim_x_f(t+1,i) = sol_mat.hx(i,:) * sim_x_f(t,:)' + sim_shocks(t,:) * eta(i,:)';
                    sim_x_s(t+1,i) = sol_mat.hx(i,:) * sim_x_s(t,:)' + 1/2 * sim_x_f(t,:) * squeeze(sol_mat.hxx(i,:,:)) * sim_x_f(t,:)' + 1/2 * sol_mat.hss(i);
                end
            end
        end

        sim_data = [sim_x_f + sim_x_s, sim_y];
    else
        for t = 1:nsim+10000
            for i = 1:ny
                sim_y(t,i) = sol_mat.gx(i,:) * sim_x(t,:)';
            end

            if t < nsim+10000
                for i = 1:nx
                    sim_x(t+1,i) = sol_mat.hx(i,:) * sim_x(t,:)' + sim_shocks(t,:) * eta(i,:)';
                end
            end
        end

        sim_data = [sim_x, sim_y];
    end
end
