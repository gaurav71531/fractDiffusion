import numpy as np
from math import sin, cos, tan, atan, pi
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.io import loadmat, savemat
from multiprocessing import Pool, cpu_count
import dill

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def get_abs_moment(x, delta):
    with np.errstate(divide='ignore'):
        return np.mean(np.power(np.abs(x), delta), axis=1)

def get_signed_abs_moment(x, delta):
    with np.errstate(divide='ignore'):
        return np.mean(np.multiply(np.sign(x), np.power(np.abs(x), delta)), axis=1)


class alpha_estimator:
    def __init__(self, a_min = 0.1, a_max = 2, b_min = 0.1, b_max = 1, use_parallel=False):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.delta_bound = a_min/5
        self.use_parallel = use_parallel
        self.is_feasible = True
        
    def fit(self, x, T, b_by_a, t_by_a):
        
        def get_abs_moment(x, delta):
            with np.errstate(divide='ignore'):
                return np.mean(np.power(np.abs(x), delta), axis=1)
        
        def linear_fit_coeff(x, T, delta_, b_by_a, t_by_a):
            abs_mom_ = get_abs_moment(x, delta_)
            T_use = np.power(T, b_by_a*delta_)
            T_use = T_use[:,np.newaxis]
            c_temp_ = np.linalg.lstsq(T_use, abs_mom_, rcond=None)[0]
            return c_temp_ * cos(delta_*pi/2) * gamma(1 + b_by_a*delta_) \
                        * gamma(1 - delta_) / cos(pi*t_by_a*delta_/2)
        
        def get_alpha_and_D_(x, y, bounds, x0):
            def func(x, a, D):
                return np.divide(np.multiply(np.power(D, x/a), pi*x/a), np.sin(pi*x/a))
            
            return curve_fit(func, x, y, bounds=bounds, p0=x0)
        
        lb = [np.max([self.a_min, self.b_min/b_by_a]), 0]
        ub = [np.min([self.a_max, self.b_max/b_by_a, 2/(1 + np.abs(t_by_a))]), 100]
        Delta = np.linspace(self.delta_bound, -np.min([self.delta_bound, 1]), 36)

        if lb[0] > ub[0]:
            print('lb:', lb)
            print('ub:', ub)
            self.is_feasible = False
            return self
        
        if self.use_parallel:
            num_processes = cpu_count()
            coeff = []
            
            with Pool(processes=num_processes) as pool:            
                for i in range(36):
                    coeff.append(apply_async(pool, linear_fit_coeff, (x, T, Delta[i], b_by_a, t_by_a)))
                coeff = [p.get() for p in coeff]
            coeff = np.squeeze( np.array(coeff))
            
            # print('time taken in coeff = %f'%(time.time()-ti))
            results_ = []
            num_trials = 500
            alpha_init = np.linspace(lb[0], ub[0], num_trials)
            D_init = np.ones((num_trials,))
            x0_all = [[alpha_init[i], D_init[i]] for i in range(num_trials)]
            
            with Pool(processes=num_processes) as pool:
                for i in range(num_trials):
                    results_.append(apply_async(pool, get_alpha_and_D_, (Delta, coeff, (lb, ub), x0_all[i])))
                results_ = [p.get() for p in results_]
            i_min_variance_alpha = np.argmin([x[1][0,0] for x in results_])
            estimation_results = results_[i_min_variance_alpha][0]
        else:
            coeff = np.empty_like(Delta)
            for i in range(np.size(Delta)):
                coeff[i] = linear_fit_coeff(x, T, Delta[i], b_by_a, t_by_a)
            x0 = [(lb[0] + ub[0])/2, 1]
            estimation_results = get_alpha_and_D_(Delta, coeff, (lb, ub), x0)[0]
        self.alpha_hat = estimation_results[0]
        self.D_hat = estimation_results[1]
        return self


def estimate_frac_diff_absm(x, T):

    x = x[1:,:] # ignoring first value as the trajectories might start from x(t) = 0
    T = np.squeeze(T)
    T = T[1:]
    delta_p = 0.001
    moment_abs_delta_p = get_abs_moment(x, delta_p)
    signed_moment_delta_p = get_signed_abs_moment(x, delta_p)
    mdl_abs_mom_p = polyfit(np.log(T), np.log(moment_abs_delta_p), 1)
    beta_by_alpha_p = mdl_abs_mom_p[1]/delta_p
    theta_by_alpha_p = 2/(pi*delta_p) * atan(-np.mean(np.divide(signed_moment_delta_p, 
                                                moment_abs_delta_p))
                                                /((1 + cos(pi*delta_p))/sin(pi*delta_p)))

    delta_n = -0.001
    moment_abs_delta_n = get_abs_moment(x, delta_n)
    signed_moment_delta_n = get_signed_abs_moment(x, delta_n)
    mdl_abs_mom_n = polyfit(np.log(T), np.log(moment_abs_delta_n), 1)
    beta_by_alpha_n = mdl_abs_mom_n[1]/delta_n
    theta_by_alpha_n = 2/(pi*delta_n) * atan(-np.mean(np.divide(signed_moment_delta_n, 
                                                moment_abs_delta_n))
                                                /((1 + cos(pi*delta_n))/sin(pi*delta_n)))

    beta_by_alpha_ = (beta_by_alpha_p + beta_by_alpha_n)/2
    theta_by_alpha_ = (theta_by_alpha_p + theta_by_alpha_n)/2
    # a_min = 0.1
    # a_max = 2
    # b_min = 0.1
    # b_max = 1
    # delta_bound = a_min/5
    alpha_est = alpha_estimator(use_parallel=True).fit(x, T, beta_by_alpha_, theta_by_alpha_)

    if alpha_est.is_feasible:
        alpha_hat = alpha_est.alpha_hat
        beta_hat = beta_by_alpha_ * alpha_hat
        theta_hat = theta_by_alpha_ * alpha_hat
        D_hat = alpha_est.D_hat
    else:
        alpha_hat=0;beta_hat=0;theta_hat=0;D_hat=0

    return alpha_est.is_feasible, alpha_hat, beta_hat, theta_hat, D_hat


def estimate_frac_diff_logm(x, T):

    x = x[1:,:] # ignoring first value as the trajectories might start from x(t) = 0
    T = np.squeeze(T)
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


def get_appended_results_(fun, M, traj, num_trials = 35, num_points = 200):

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
                is_feasible, alpha_hat, beta_hat, theta_hat, D_hat = fun(x_use, T)
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

if __name__ == '__main__':
    data_raw = loadmat('data/sim_D_1.00_A_2.00_B_1.00_theta_0.00_N_30000_M_10000_L_1000.mat')
    traj_matlab = {'T':data_raw['T'], 'x':data_raw['xnt']}
    estimate_frac_diff_absm(traj_matlab['x'], traj_matlab['T'])