% clc
% clear;
% close all
dataDirName = 'data';
resultsDirName = 'results';
% alpha_ = 2;
% beta_ = 0.5;
% theta_ = 0;
% D_ = 1;
% fName = sprintf('sim_D_%1.2f_A_%1.2f_B_%1.2f_theta_%1.2f_N_30000_M_10000_L_1000.mat',D_, alpha_, beta_, theta_);

alpha_ = 2;H_ = 0.25; beta_ = 2*H_; theta_ = 0;D_ = 2;
fName = sprintf('sim_fBm_H_%1.2f_L_1000_M_10000.mat',H_);
fList = {fName};
% fList = {'sim_D_1.00_A_2.00_B_0.50_theta_0.00_N_30000_M_10000_L_1000.mat',...
%     };
% fList = {'sim_D_1.00_A_2.00_B_1.00_theta_0.00_N_30000_M_10000_L_1000.mat',...
%     };
% fList = {'sim_D_1.00_A_2.00_B_1.00_theta_0.00_N_30000_M_10000_L_1000.mat',...
%     'sim_D_2.00_A_2.00_B_1.00_theta_0.00_N_30000_M_10000_L_1000.mat'
%     };
%
for nam=1:length(fList)
    load(fullfile(dataDirName, fList{nam}));
    %
%     MMM = floor(logspace(1,4,70));
    MMM = floor(logspace(1,4,35));
%     MMM = fliplr(MMM);
    POINT = 200;
    ALPHA = zeros(length(MMM), POINT);
    BETA = zeros(length(MMM), POINT);
    THETA = zeros(length(MMM), POINT);
    DiffCoeff = zeros(length(MMM), POINT);
    BOA = zeros(length(MMM), POINT);
    TOA = zeros(length(MMM), POINT);
    t_total = tic;
    for k = 1:POINT
        ti = tic;
        for p =1:length(MMM)
%             ti = tic;
    %
            MM = MMM(p);
            
%             XNT = xnt(:,ceil(rand(1,MM)*M));
            XNT = xnt(:,randperm(M, MM));
            %
%             format long
            e = 0.001;

            delta = -e;
            Moment = sum((abs(XNT')).^delta)/MM;
            mdl= polyfit(log(T(2:end))',log(Moment(2:end))',1);
            B1 = mdl(1)/delta;

            MomentS = sum(sign(XNT').*(abs(XNT')).^delta)/MM;
            TA1 = 2/(pi*delta) * atan(...
                                    -mean(MomentS(2:end)./Moment(2:end))...
                                    /((1+cos(pi*delta))/sin(pi*delta)));
            delta = +e;
            Moment = sum((abs(XNT')).^delta)/MM;

            mdl= polyfit(log(T(2:end))',log(Moment(2:end))',1);
            B2 = mdl(1)/delta;

            MomentS = sum(sign(XNT').*(abs(XNT')).^delta)/MM;
            TA2 = 2/(pi*delta) * atan(...
                                    -mean(MomentS(2:end)./Moment(2:end))...
                                    /((1+cos(pi*delta))/sin(pi*delta)));
            B = (B1+B2)/2; %B = Beta/Alpha;
            BOA(p,k) = B;
            TA = (TA1+TA2)/2;
            TOA(p,k) = TA;
            %%%%%%% alpha estimate module%%%%%
            Alpmin = 0.1;
            Alpmax = 2;
            Betmax = 1;
            Betmin = 0.1;

            E = Alpmin/5; %E = Alpha/5;%for theta

            % lb = [max([+Alpmin, Betmin/B          ])  ];
            % ub = [min([+Alpmax, Betmax/B, 2/(1+abs(TA))])  ];
            % x0 = (lb+ub)/2;
            % 
            % if lb>ub
            lb = [max([+Alpmin, Betmin/B          ])       ,0];
            ub = [min([+Alpmax, Betmax/B, 2/(1+abs(TA))])  ,100];
            x0 = [(lb(1)+ub(1))/2,1];

            if lb(1)>ub(1)
                continue;
            end

            Delta = linspace(+E,-min(E,1),36);
            A = zeros(1, length(Delta));
            C = zeros(1, length(Delta));
            for i = 1:size(Delta,2)
                delta = Delta(i);

                Moment = sum((abs(XNT')).^delta)/MM;
                dlm = fitlm((T(2:end).^(B*delta))',Moment(2:end)','Intercept',false);
                A(i) = dlm.Coefficients.Estimate;
                C(i) = A(i) * (cos(delta*pi/2)*gamma(1+B*delta)*gamma(1-delta))...
                    / cos(delta*pi*TA/2);
            end
            %
            xdata = Delta;
            ydata = C;

            fun = @(x,xdata)(x(2).^(xdata./x(1)).*((pi*xdata./x(1))./ sin(pi*xdata./x(1))));
            problem = createOptimProblem('lsqcurvefit','x0',x0,'objective',fun,...
                'lb',lb,'ub',ub,'xdata',xdata,'ydata',ydata);

            ms = MultiStart('UseParallel',true, ...
                        'Display','off');%('PlotFcns',@gsplotbestf);
            [x,errormulti] = run(ms,problem,500);

%             optim_options = optimoptions('lsqcurvefit','Display', 'off');
%             x = lsqcurvefit(fun,x0,xdata,ydata,lb,ub, optim_options);

            alphaHat = x(1);
            DiffCoeff(p,k) = x(2);
            
% % % % % % % % % % % % % % % % % % %             
            ALPHA(p,k) = alphaHat;
            BETA (p,k) = B*alphaHat;
            THETA(p,k) = TA*alphaHat;
%             fprintf('Completed for (p,k) = (%d, %d), time taken = %f\n', p,k, toc(ti));
        end
        fprintf('Completed for k = %d, time taken = %f\n', k, toc(ti));
        
        if ~mod(k, 20) % save threshold
            save(fullfile(resultsDirName, ['FIRST_',fList{nam}]), ...
                'ALPHA', 'BETA', 'THETA', 'DiffCoeff', 'BOA', 'TOA');
        end
    end
    finalStr = sprintf('total time taken = %f\n', toc(t_total));
    fprintf(finalStr);
    
%     clear('Moment','xnt','XNT');
    save(fullfile(resultsDirName, ['FIRST_',fList{nam}]), ...
        'ALPHA', 'BETA', 'THETA', 'DiffCoeff', 'BOA', 'TOA');
end