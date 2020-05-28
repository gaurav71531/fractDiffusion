import numpy as np
from scipy.stats import levy_stable
from math import tan, cos, pi
import dill
from time import time
from multiprocessing import Pool, cpu_count
import pickle as pk
from scipy.io import savemat
import os

def make_dir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))

def get_levy_stable(alpha, beta, delta, gamma, size_):
    return levy_stable.rvs(alpha, beta, delta, gamma, size = size_)


def fractional_diffusion(alpha_, beta_, theta_, D_, L, M, 
                        use_parallel = False, do_save = False):
    L = int(L)
    M = int(M)
    xnt = np.zeros((L, M))
    L_temp = 20*L
    tau = 1e-5
    c_alpha_ = np.power(D_*tau, 1/alpha_)
    c_beta_ = np.power(tau, 1/beta_)

    x_alpha_ = alpha_
    x_beta_ = - tan(theta_*pi/2) / tan(alpha_*pi/2)
    x_gamma_ = c_alpha_ * np.power(cos(theta_*pi/2), 1/alpha_)
    x_delta_ = - x_gamma_ * tan(theta_*pi/2)

    t_alpha_ = beta_
    t_beta_ = 1
    t_gamma_ = c_beta_ * np.power(cos(beta_*pi/2), 1/beta_)
    t_delta_ = t_gamma_ * tan(beta_*pi/2)
    M_thres = 200
    num_processes = cpu_count()
    ti = time()
    if M>M_thres:
        num_M = int(np.floor(M/M_thres))
        if use_parallel:
            pool = Pool(processes=num_processes)
            results_dt = []
            results_dx = []
            for i in range(num_M):
                results_dt.append(apply_async(pool, get_levy_stable, (t_alpha_, t_beta_, t_delta_, t_gamma_, (L_temp, M_thres))))
            results_dt = [p.get() for p in results_dt]
            dt = np.concatenate(results_dt, axis=1)

            for i in range(num_M):
                results_dx.append(apply_async(pool, get_levy_stable, (x_alpha_, x_beta_, x_delta_, x_gamma_, (L_temp, M_thres))))
            results_dx = [p.get() for p in results_dx]
            dx = np.concatenate(results_dx, axis=1)
        else:
            dt = np.zeros((L_temp, M))
            dx = np.zeros((L_temp, M))
            for i in range(num_M):
                dt_temp = levy_stable.rvs(t_alpha_, t_beta_, t_delta_, t_gamma_, size = (L_temp, M_thres))
                dt[:,i*M_thres:(i+1)*M_thres] = dt_temp

                dx_temp = levy_stable.rvs(x_alpha_, x_beta_, x_delta_, x_gamma_, size = (L_temp, M_thres))
                dx[:,i*M_thres:(i+1)*M_thres] = dx_temp
    else:
        dt = levy_stable.rvs(t_alpha_, t_beta_, t_delta_, t_gamma_, size = (L_temp, M))
        dx = levy_stable.rvs(x_alpha_, x_beta_, x_delta_, x_gamma_, size = (L_temp, M))

    dt_sum = np.concatenate((np.zeros((1, M)), np.cumsum(dt, axis=0)))
    dx_sum = np.concatenate((np.zeros((1, M)), np.cumsum(dx, axis=0)))
    T = np.linspace(0, np.min(dt_sum[-1,:]), L)

    nt = np.zeros((np.size(T), M), dtype=int)
    for i in range(M):
        jj = 0
        looping_complete = False
        for k in range(1+L_temp):
            while(True):
                if dt_sum[k,i]>=T[jj]:
                    nt[jj,i] = k
                    jj += 1
                    if jj >= np.size(T):
                        looping_complete = True
                        break
                else:
                    break
            if looping_complete:
                break
    for i in range(M):
        xnt[:,i] = dx_sum[nt[:,i],i]

    if do_save:
        data_dir_name = 'data'
        make_dir(data_dir_name)
        file_name = 'sim_frac_diff_a_%1.2f_b_%1.2f_t_%1.2f_D_%1.2f_L_%d_M_%d.p'%(alpha_, beta_, theta_, D_, L, M)
        pk.dump({'x':xnt, 'T':T}, open(os.path.join(data_dir_name, file_name), 'wb'))
        savemat(open(os.path.join(data_dir_name, file_name[:-1]+'mat'), 'wb'), {'xnt':xnt, 'T':T})

    return {'x':xnt, 'T':T}

def main():
    M = 1e4
    L = 1e3
    alpha_ = 2
    beta_ = 0.5
    theta_ = 0
    D_ = 1
    trajectories = fractional_diffusion(alpha_, beta_, theta_, D_, L, M, use_parallel = True)
    return 1


if __name__ == '__main__':
    main()