function [ALPHA, BETA, THETA, D_] = fract_diff_est_absm(XNT, T)

    e = 0.001;
    delta = -e;
    Moment = mean((abs(XNT)).^delta, 2);
    mdl= polyfit(log(T(2:end))',log(Moment(2:end))',1);
    B1 = mdl(1)/delta;

    MomentS = mean(sign(XNT).*(abs(XNT)).^delta, 2);
    TA1 = 2/(pi*delta) * atan(...
                            -mean(MomentS(2:end)./Moment(2:end))...
                            /((1+cos(pi*delta))/sin(pi*delta)));
    delta = +e;
    Moment = mean((abs(XNT)).^delta, 2);

    mdl= polyfit(log(T(2:end))',log(Moment(2:end))',1);
    B2 = mdl(1)/delta;

    MomentS = mean(sign(XNT).*(abs(XNT)).^delta, 2);
    TA2 = 2/(pi*delta) * atan(...
                            -mean(MomentS(2:end)./Moment(2:end))...
                            /((1+cos(pi*delta))/sin(pi*delta)));
    B = (B1+B2)/2;
    TA = (TA1+TA2)/2;
    %%%%%%% alpha estimate module%%%%%
    Alpmin = 0.1;
    Alpmax = 2;
    Betmax = 1;
    Betmin = 0.1;

    E = Alpmin/5; %E = Alpha/5;%for theta

    lb = [max([+Alpmin, Betmin/B          ])       ,0];
    ub = [min([+Alpmax, Betmax/B, 2/(1+abs(TA))])  ,100];
    x0 = [(lb(1)+ub(1))/2,1];

    if lb(1)>ub(1)
        fprintf('bounds determination error, please check...\n');
    end

    Delta = linspace(+E,-min(E,1),36);
    A = zeros(1, length(Delta));
    C = zeros(1, length(Delta));
    for i = 1:size(Delta,2)
        delta = Delta(i);

        Moment = mean((abs(XNT)).^delta, 2);
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
                'Display','off');
    [x,~] = run(ms,problem,500);

    D_ = x(2);
    ALPHA = x(1);
    BETA = B*ALPHA;
    THETA = TA*ALPHA;