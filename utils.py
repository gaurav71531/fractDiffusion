import numpy as np
from math import sin, cos, tan, atan, pi
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
import dill

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def estimate_frac_diff_1(x, T):

    def get_abs_moment(x, delta):
        return np.mean(np.power(np.abs(x), delta), axis=1)
    def get_signed_abs_moment(x, delta):
        return np.mean(np.multiply(np.sign(x), np.power(np.abs(x), delta)), axis=1)

    x = x[1:,:] # ignoring first value as the trajectories might start from x(t) = 0
    T = T[1:]
    delta_p = 0.001
    moment_abs_delta_p = get_abs_moment(x, delta_p)
    signed_moment_delta_p = get_signed_abs_moment(x, delta_p)
    mdl_abs_mom_p = polyfit(np.log(T), np.log(moment_abs_delta_p), 1)
    beta_by_alpha_p = mdl_abs_mom_p[1]
    theta_by_alpha_p = 2/(pi*delta_p) * atan(-np.mean(np.divide(signed_moment_delta_p, 
                                                moment_abs_delta_p))
                                                /((1 + cos(pi*delta_p))/sin(pi*delta_p)))

    delta_n = -0.001
    moment_abs_delta_n = get_abs_moment(x, delta_n)
    signed_moment_delta_n = get_signed_abs_moment(x, delta_n)
    mdl_abs_mom_n = polyfit(np.log(T), np.log(moment_abs_delta_n), 1)
    beta_by_alpha_n = mdl_abs_mom_n[1]
    theta_by_alpha_n = 2/(pi*delta_n) * atan(-np.mean(np.divide(signed_moment_delta_n, 
                                                moment_abs_delta_n))
                                                /((1 + cos(pi*delta_n))/sin(pi*delta_n)))

    beta_by_alpha_ = (beta_by_alpha_p + beta_by_alpha_n)/2
    theta_by_alpha_ = (theta_by_alpha_p + theta_by_alpha_n)/2

    a_min = 0.1
    a_max = 2
    b_min = 0.1
    b_max = 1
    delta_bound = a_min/5

    lb = [np.max([a_min, b_min/beta_by_alpha_]), 0]
    ub = [np.min([a_max, b_max/beta_by_alpha_, 2/(1 + np.abs(theta_by_alpha_))]), 100]
    x0 = [(lb[0] + ub[0])/2, 1]

    Delta = np.linspace(delta_bound, -np.min(delta_bound, 1), 36)
    




def estimate_frac_diff_2(x, T):

    def get_abs_moment(x, delta):
        return np.mean(np.power(np.abs(x), delta), axis=1)
    def get_signed_abs_moment(x, delta):
        return np.mean(np.multiply(np.sign(x), np.power(np.abs(x), delta)), axis=1)

    x = x[1:,:] # ignoring first value as the trajectories might start from x(t) = 0
    T = T[1:]
    delta_p = 0.001
    moment_abs_delta_p = get_abs_moment(x, delta_p)
    signed_moment_delta_p = get_signed_abs_moment(x, delta_p) 
    theta_by_alpha_p = 2/(pi*delta_p) * atan(-np.mean(np.divide(signed_moment_delta_p, 
                                                moment_abs_delta_p))
                                                /((1 + cos(pi*delta_p))/sin(pi*delta_p)))

    delta_n = -0.001
    moment_abs_delta_n = get_abs_moment(x, delta_n)
    signed_moment_delta_n = get_signed_abs_moment(x, delta_n)
    theta_by_alpha_n = 2/(pi*delta_n) * atan(-np.mean(np.divide(signed_moment_delta_n, 
                                                moment_abs_delta_n))
                                                /((1 + cos(pi*delta_n))/sin(pi*delta_n)))

    log_abs_x = np.log(np.abs(x))
    mean_log_abs_x = np.mean(log_abs_x, axis=1)

    mdl_log_abs = polyfit(np.log(T), mean_log_abs_x, 1)
    beta_by_alpha_ = mdl_log_abs[1]
    theta_by_alpha_ = (theta_by_alpha_p + theta_by_alpha_n)/2

    var_log_abs = np.var(log_abs_x, axis=1, ddof=1)
    var_log_abs_use = np.mean(var_log_abs)
    param_temp = ((var_log_abs_use + (pi*theta_by_alpha_/2)**2)*(6/pi**2) - 0.5
                    + beta_by_alpha_**2)/2
    if param_temp<0:
        is_feasible = False
        alpha_hat = 0;beta_hat = 0;theta_hat = 0;D_hat = 0
    else:
        is_feasible = True
        alpha_hat = np.power(param_temp, -0.5)
        D_hat = np.exp(alpha_hat * (mdl_log_abs[0] - 0.577*(beta_by_alpha_-1)))
        beta_hat = beta_by_alpha_ * alpha_hat
        theta_hat = theta_by_alpha_ * alpha_hat

    return is_feasible, alpha_hat, beta_hat, theta_hat, D_hat


def get_appended_results_(M, traj, num_trials = 35, num_points = 200):

    def get_trials_(num_trials, M, M_try, x, T):
    
        alpha_trial = np.zeros((num_trials,))
        beta_trial = np.zeros((num_trials,))
        theta_trial = np.zeros((num_trials,))
        D_trial = np.zeros((num_trials,))
        for t in range(num_trials):
            max_try = 20
            ctr = 0
            M_use = M_try[t]
            while(ctr<max_try):
                random_traj_index = np.random.choice(M, M_use).astype('int')
                x_use = x[:,random_traj_index]
                is_feasible, alpha_hat, beta_hat, theta_hat, D_hat = estimate_frac_diff_2(x_use, T)
                if not is_feasible:
                    ctr += 1
                    continue
                alpha_trial[t] = alpha_hat
                beta_trial[t] = beta_hat
                theta_trial[t] = theta_hat
                D_trial[t] = D_hat
                break
        return alpha_trial, beta_trial, theta_trial, D_trial

    M = int(M)
    M_try = np.floor(np.logspace(0, np.log10(M), num_trials)).astype('int')
    alpha_try = np.zeros((num_trials, num_points))
    beta_try = np.zeros((num_trials, num_points))
    theta_try = np.zeros((num_trials, num_points))
    D_try = np.zeros((num_trials, num_points))
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)
    results = []
    for p in range(num_points):
        results.append(apply_async(pool, get_trials_, (num_trials, M, M_try, traj['x'], traj['T'])))
    results = [p.get() for p in results]
    
    return results