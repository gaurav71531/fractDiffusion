%% Simulation for trajectories from param tuple

alpha = 2;
beta = 0.5;
theta = 0;
D = 1;
time_len = 1000;
num_traj = 1e4;
save_to_directory = false;
[XNT, T] = gen_sim_data(alpha, beta, theta, D, time_len, num_traj, save_to_directory);

% estimate paramameters using abs mom algorithm (Algorithm-1 of the paper)
[alpha_hat1, beta_hat1, theta_hat1, D_hat1] = fract_diff_est_absm(XNT, T);

% estimate parameters using log mom algorithm (Algorithm-2 of the paper)
[alpha_hat2, beta_hat2, theta_hat2, D_hat2] = fract_diff_est_logm(XNT, T);

fprintf('The estimated values of the parameters are as follows:\n');
fprintf('For Algorithm-1 ...\n');
fprintf('alpha = %f, beta = %f, theta = %f, D = %f\n', alpha_hat1, beta_hat1, theta_hat1, D_hat1);
fprintf('For Algorithm-2 ...\n');
fprintf('alpha = %f, beta = %f, theta = %f, D = %f\n', alpha_hat2, beta_hat2, theta_hat2, D_hat2);

%% Simulation for estimated parameters variance across different number of trajectories

alpha = 2;
beta = 0.5;
theta = 0;
D = 1;
time_len = 1000;
num_points = 200;
num_traj_total = 1e4;
[XNT, T] = gen_sim_data(alpha, beta, theta, D, time_len, num_traj_total, false);
num_traj = floor(logspace(1, 4, 35));
ALPHA_1 = zeros(length(num_traj), num_points);
BETA_1 = zeros(length(num_traj), num_points);
THETA_1 = zeros(length(num_traj), num_points);
D_1 = zeros(length(num_traj), num_points);

ALPHA_2 = zeros(length(num_traj), num_points);
BETA_2 = zeros(length(num_traj), num_points);
THETA_2 = zeros(length(num_traj), num_points);
D_2 = zeros(length(num_traj), num_points);

max_try = 20;
for p = 1:num_points
    for t = 1:length(num_traj)
        successful_run = false;
        ctr = 1;
        while(ctr<max_try)
            try
                xnt_use = XNT(:,randperm(num_traj_total, num_traj(t)));
                [ALPHA_1(t, p), BETA_1(t, p), THETA_1(t, p), D_1(t, p)] = fract_diff_est_absm(xnt_use, T);
                [ALPHA_2(t, p), BETA_2(t, p), THETA_2(t, p), D_2(t, p)] = fract_diff_est_logm(xnt_use, T);        
                fprintf('Completed for tuple (p, t) = (%d, %d)\n', p, t);
                successful_run = true;
            catch

            end
            if successful_run
                break
            end
            ctr = ctr + 1;
            if ctr == max_try
                fprintf('Error not terminated due to less number of trajectories, try to use more...\n');
            end
        end
    end
end

params_to_plot = {ALPHA_1, BETA_1, THETA_1, D_1;
    ALPHA_2, BETA_2, THETA_2, D_2};
params_name = {'alpha', 'beta', 'theta', 'D'};
params_orig = [alpha, beta, theta, D];

for i = 1:length(params_to_plot)
    
    figure;
    x = linspace(1, 4, 35);
    mean_param = mean(params_to_plot{1,i},2);
    std_param = std(params_to_plot{1,i}, 0, 2);

    y_upper = mean_param + std_param;
    y_lower = mean_param - std_param;
    fill_x = [x, fliplr(x)];
    inBetween = [y_upper; flipud(y_lower)]';

    plot(x, mean_param, 'b', 'linewidth', 1.5);
    hold on;
    fill(fill_x, inBetween, 'b', 'FaceAlpha', 0.25, 'LineStyle', 'none',...
            'HandleVisibility','off');
    hold on;
    
    mean_param = mean(params_to_plot{2,i},2);
    std_param = std(params_to_plot{2,i}, 0, 2);

    y_upper = mean_param + std_param;
    y_lower = mean_param - std_param;
    fill_x = [x, fliplr(x)];
    inBetween = [y_upper; flipud(y_lower)]';

    plot(x, mean_param, 'k', 'linewidth', 1.5);
    hold on;
    fill(fill_x, inBetween, 'k', 'FaceAlpha', 0.25, 'LineStyle', 'none',...
        'HandleVisibility','off');
    
    hold on;
    plot(x, params_orig(i)* ones(size(x)), 'r--', 'linewidth', 1.5);
    grid;
    ylabel(params_name{i});
    xlabel('log(N)');
    legend({'Algorithm-1', 'Algorithm-2', 'original'})
end