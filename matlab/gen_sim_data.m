function [xnt, T] = gen_sim_data(Alpha, Beta, theta, D, time_len, num_traj, save_in_dir)

% Alpha between 0 and 2
% Beta between 0 and 1
% theta between -1 to 1 of (min([Alpha,2-Alpha,2-Beta]))
% D between 0 to inf

dataDirName = 'data';
if save_in_dir
    if ~exist(dataDirName, 'dir')
        mkdir(dataDirName)
    end
end

%      
L = time_len; % length
N = 30*L;
%     M = 10000; % # of trajectories
M = num_traj;
ta = 1e-5;
%
C_Alpha = (D*ta)^(1/Alpha);
C_Beta = ta^(1/Beta);
% reisz-feller
X_Alpha = Alpha;
X_Beta = - tan(theta*pi/2) / tan(Alpha*pi/2);
X_gam = C_Alpha * (cos(theta*pi/2))^(1/Alpha);
X_delta = - X_gam * tan(theta*pi/2);

X_pdf = makedist('Stable','Alpha',X_Alpha,'Beta',X_Beta,'gam',X_gam,'delta',X_delta);
% caputo
T_Alpha = Beta;
T_Beta = 1;
T_gam = C_Beta * (cos(Beta*pi/2))^(1/Beta);
T_delta = T_gam * tan(Beta*pi/2);

T_pdf = makedist('Stable','Alpha',T_Alpha,'Beta',T_Beta,'gam',T_gam,'delta',T_delta);
%
tic
M_thres = min(500, M);
num_M = floor(M/M_thres);
dt = zeros(N, M);
for i = 1:num_M
    dt_temp = random(T_pdf, N, M_thres);
    dt(:,(i-1)*M_thres+1:i*M_thres) = dt_temp;
end
fprintf('time taken for dt samples = %f\n', toc);
tsum = [zeros(1,M);cumsum(dt)];
T = linspace(0,(min(max(tsum))),L);
T = T';
tic
dx = zeros(N, M);
for i = 1:num_M
    dx_temp = random(X_pdf, N, M_thres);
    dx(:,(i-1)*M_thres+1:i*M_thres) = dx_temp;
end
fprintf('time taken for dx samples = %f\n', toc);
xsum = [zeros(1,M);cumsum(dx)];

nt = zeros(length(T), M);
for i = 1:M
    jj = 1;
    loopingComplete = false;
    for k = 1:size(tsum,1)
        while(true)
            if tsum(k,i) >= T(jj)
                nt(jj,i) = k;
                jj = jj + 1;
                if jj > length(T)
                    loopingComplete = true;
                    break;
                end
            else
                break;
            end
        end
        if loopingComplete, break;end
    end
end
for i = 1:M
    xnt(:,i) = xsum(nt(:,i) ,i);
end
% figure;plot(xnt(:,5));
if save_in_dir
    fileName = sprintf('sim_D_%1.2f_A_%1.2f_B_%1.2f_theta_%1.2f_N_%d_M_%d_L_%d.mat',D, Alpha, Beta, theta, N, M, L);
    save(fullfile(dataDirName, fileName), 'M','xnt','T','Alpha','Beta','theta','D');
end