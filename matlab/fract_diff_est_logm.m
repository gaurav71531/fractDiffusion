function [ALPHA, BETA, THETA, D_] =  fract_diff_est_logm(XNT, T)
    %
    e = 0.001;

    delta = -e;
    Moment_deltan = mean((abs(XNT)).^delta, 2);
    MomentS_deltan = mean(sign(XNT).*(abs(XNT)).^delta, 2);

    TA1 = 2/(pi*delta) * atan(...
                            -mean(MomentS_deltan(2:end)./Moment_deltan(2:end))...
                            /((1+cos(pi*delta))/sin(pi*delta)));

    delta = +e;
    Moment_deltap = mean((abs(XNT)).^delta, 2);
    MomentS_deltap = mean(sign(XNT).*(abs(XNT)).^delta, 2);

    TA2 = 2/(pi*delta) * atan(...
                            -mean(MomentS_deltap(2:end)./Moment_deltap(2:end))...
                            /((1+cos(pi*delta))/sin(pi*delta)));

    log_abs_xnt = log(abs(XNT));
    mean_log_abs_xnt = mean(log_abs_xnt,2);

    mdl_log_abs = polyfit(log(T(2:end)), mean_log_abs_xnt(2:end),1);
    beta_by_alpha_ = mdl_log_abs(1);
    
    if beta_by_alpha_ < 0
        fprintf('Negative beta by alpha detected...\n');
    end

    theta_by_alpha_ = (TA1+TA2)/2;

    var_log_abs_enm = var(log_abs_xnt, 0, 2);


    var_log_abs_enm_use = mean(var_log_abs_enm(2:end));
    param_temp = ((var_log_abs_enm_use + (pi*theta_by_alpha_/2)^2)*(6/pi^2) - 0.5 ...
        + beta_by_alpha_^2)/2;
    if param_temp<0
        fprintf('negative issue detected at (p,k) = (%d,%d)\n', p, k);
    end
    alpha_hat = param_temp^(-0.5);
    alpha_hat = min(2, alpha_hat);
    D_ = exp(alpha_hat*(mdl_log_abs(2) - 0.577*(beta_by_alpha_-1)));
    ALPHA = alpha_hat;
    BETA = beta_by_alpha_*alpha_hat;
    THETA = theta_by_alpha_*alpha_hat;  