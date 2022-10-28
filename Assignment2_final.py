
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as st
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sns
from scipy.stats import norm
from scipy.special import ndtri
from scipy import optimize

#a)/b)
exchange = pd.read_excel("/Users/stephanweenk/Downloads/VU Amsterdam/Econometrie/Time Series Models/Syllabus, exercises, assignments/Assignment 2/SvData.xlsx", delimiter = ';')
exchange = exchange/100
x = np.log((exchange-np.mean(exchange))**2)
Time = len(x)
Y = x.values

exchange_dem = exchange-np.mean(exchange)

# State smoother
def state_smoother_SV(a, P, v, F, y, T, Z, K):
    n = len(y)
    # vectors to be filled
    Alpha_hat = np.zeros(n)  # smoothed state
    L = np.zeros(n)
    r = np.zeros(n + 1)  # state smoothing cumulant
    N = np.zeros(n + 1)  # State variance cumulant
    V = np.zeros(n)  # smoothed state variance

    for t in range(0, n):
        if not np.isnan(y[t]):
            L[t] = T - K[t] * Z
        else:
            L[t] = 1

    # Reverse recursion for cumulants
    for t in range(n - 1, -1, -1):
        if not np.isnan(y[t]):
            r[t] = Z * v[t] / (F[t]) + L[t] * r[t + 1]
            N[t] = Z ** 2 / F[t] + (L[t] ** 2) * N[t + 1]
        else:
            r[t] = r[t + 1]
            N[t] = (1 / F[t]) + (L[t] ** 2) * N[t + 1]

    # variance and state smoothing
    for t in range(0, n):
        V[t] = P[t] - (P[t] ** 2) * N[t]
        Alpha_hat[t] = a[t] + P[t] * r[t]

    return Alpha_hat, V, r, N, L


# =============================================================================
# Plot raw data
plt.figure(1, figsize=(15,4))
plt.plot(np.arange(len(exchange)), exchange_dem/100, linewidth=0.75, color = 'grey')
plt.hlines(y=0, xmin=0, xmax=len(exchange), linewidth=1)
plt.show()

# Plot transformed data to make the SV model linear
plt.figure(1, figsize=(15,4))
plt.scatter(np.arange(len(x)), x, s = 4, color = 'black')
#plt.hlines(y=0, xmin=0, xmax=len(x), linewidth=1)
plt.show()

log_ret_df = pd.DataFrame(exchange)
stats = log_ret_df.describe()
stats.loc[8] = log_ret_df.median()
stats.loc[9] = log_ret_df.skew()
stats.loc[10]= log_ret_df.kurtosis()
stats.rename(index={8: 'median',9: 'skewness',10: 'kurtosis'})


# c)
# Kalman filter
def Kalman_Filter_SV(y, H, Q, phi, R, Z, a_1, P_1, d, omega):  ### Note that time is len(x)
    # Construct vectors
    n = len(x)
    a = np.zeros(n)
    a[0] = a_1
    P = np.zeros(n)
    P[0] = P_1
    v = np.zeros(n)
    F = np.zeros(n)
    K = np.zeros(n)

    # Compute the values
    for i in range(0, n - 1):
        if not np.isnan(y[i]):
            v[i] = y[i] - Z * a[i] - d  ### allow for intercept
            if isinstance(H, float):
                F[i] = Z * P[i] * Z + H
            else:
                F[i] = Z * P[i] * Z + H
            K[i] = phi * P[i] * Z * (1 / F[i])
            a[i + 1] = omega + phi * a[i] + K[i] * v[i]
            P[i + 1] = phi * P[i] * phi + R * Q * R - K[i] * F[i] * K[i]
        else:
            v[i] = np.nan
            F[i] = 10 ** 9
            K[i] = 0
            a[i + 1] = a[i]
            P[i + 1] = P[i] + Q

    v[-1] = y[len(y) - 1] - Z * a[-1]
    F[-1] = P[-1] + H
    return a, P, v, F, K


def loglik_SV(theta, y):
    # Construct known and theta
    H = np.pi ** 2 / 2
    Z = 1
    d = -1.27
    omega = theta[0]  ### c
    phi = theta[1]  ### T
    R =  1
    Q = theta[2]### sigma_etasquared
    a_1 = np.mean(y)
    P_1 = Q / (1 - phi ** 2)

    a, P, v, F, K = Kalman_Filter_SV(y, H, Q, phi, R, Z, a_1, P_1, d, omega)
    l = (-1 / 2) * np.log(2 * np.pi) - (1 / 2) * np.log(F) - (1 / 2) * np.square(v) / F
    llik = np.mean(l)

    return -llik


# QML SV model
options = {'eps': 1e-07,  # argument convergence criteria
           'disp': True,  # display iterations
           'maxiter': 5000}  # maximum number of iterations
ML_SV = optimize.minimize(loglik_SV, np.array([0.001, 0.9, 0.5]), args=(Y), method='SLSQP', options=options)
print(ML_SV)

H = np.pi ** 2 / 2
Z = 1
d = -1.27
omega = ML_SV.x[0]  ### c
phi = ML_SV.x[1]  ### T
R =   1
Q = ML_SV.x[2]### sigma_etasquared
a_1 = np.mean(Y)
P_1 = Q / (1 - phi ** 2)

a, P, v, F, K = Kalman_Filter_SV(Y, H, Q, phi, R, Z, a_1, P_1, d, omega)

Alpha_hat, V, r, N, L = state_smoother_SV(a, P, v, F, Y, phi, Z, K)

# d)
### Figures
plt.figure(1, figsize=(15,4))
plt.plot(Alpha_hat[1:], color='black', label='$h_{t}$ smoothed')
plt.plot(a[1:], color='red', label='$h_{t}$')
plt.scatter(range(len(Y) - 1), Y[1:], s=2, color = 'black')
plt.legend()

ksi = omega / (1 - phi)
H_vec_smoothed = Alpha_hat - ksi
H_vec = a - ksi

plt.figure(1, figsize=(15,4))
plt.plot(H_vec_smoothed, color='black', label='$H_{t}$ smoothed')
plt.plot(H_vec, color='red', label='$H_{t}$')
plt.legend()

#f)
raw_data= exchange.values

def particle_filter(y, phi, Q, N, ksi):
    T= len(y)
    theta_tilde = np.zeros((N,T))
    sigma = np.zeros(N)
    w_tilde = np.zeros(N)
    w_norm = np.zeros(N)
    H_estimate = np.zeros(T)
    mu = np.mean(y)



    # Now we obtain all the theta tildes
    theta_tilde[:, 0] = np.random.normal(loc=0,
                                         scale=np.sqrt(Q / (1 - phi ** 2)), size=N)
    # and the unconditional sigma
    sigma[:] = np.exp((1/2)*(theta_tilde[:, 0]+ksi))
    # obtain w tilde
    #w_tilde[:] = (1/(sigma[:]*np.sqrt(2*np.pi)))*np.exp((-(y[0]-mu)**2)/(2*sigma[:]**2)) #### Include pdf function of normal with differing loc and scale
    w_tilde[:] = norm.pdf(y[0], loc = mu, scale = sigma[:])
    # then we normalize the weights

    w_norm[:] = w_tilde[:] / np.sum(w_tilde[:])

    # then we compute a hat, which is an estimate of H in the case of our SV model
    H_estimate[0] = np.sum(w_norm[:] * theta_tilde[:, 0])
    theta_tilde[:, 0] = np.random.choice(theta_tilde[:, 0], p=w_norm[:])



    for j in range(1, T):
        # Now we obtain all the theta tildes
        theta_tilde[:, j] = np.random.normal(loc=phi * theta_tilde[:, j - 1], scale=np.sqrt(Q), size=N)
        # and the unconditional sigma
        sigma[:] = np.exp((1/2)*(theta_tilde[:, 0]+ksi))
        # obtain w tilde
        #w_tilde[:] = (1/(sigma[:]*np.sqrt(2*np.pi)))*np.exp((-(y[j]-mu)**2)/(2*sigma[:]**2))
        w_tilde[:] = norm.pdf(y[j], loc = mu, scale = sigma[:])
        # then we normalize the weights
        w_norm[:] = w_tilde[:] / np.sum([w_tilde[:]])
        # then we compute a hat, which is an estimate of H in the case of our SV model
        H_estimate[j] = np.sum(w_norm[:] * theta_tilde[:, j])
        # and we resample the theta for the next period
        theta_tilde[:, j] = np.random.choice(theta_tilde[:, j], p=w_norm[:], replace = True)

    return H_estimate

# H_particle_vector = np.zeros((100, len(raw_data)))
# H_particle = np.zeros(len(raw_data))
#
# for i in range(0, 100):
#     H_particle_vector[i,:] = particle_filter(raw_data, phi, Q, 10000, ksi)
#
#

np.random.seed(2003)
H_particle = particle_filter(raw_data, phi, Q, 10000, ksi)

plt.figure(1, figsize=(15,4))
plt.plot(H_particle, color= 'red', label = 'Particle filter', linestyle= 'dashed')
# for i in range(0, 100):
#     plt.plot(H_particle_vector[i, :], color='red', label='Particle filter', linestyle='dashed')
plt.plot(H_vec, color= 'black', label= 'Kalman filter')
plt.legend()
plt.show()

np.random.seed(2008)
H_particle = particle_filter(raw_data, phi, Q, 10000, ksi)

plt.figure(1, figsize=(15,4))
plt.plot(H_particle, color= 'red', label = 'Particle filter', linestyle= 'dashed')
# for i in range(0, 100):
#     plt.plot(H_particle_vector[i, :], color='red', label='Particle filter', linestyle='dashed')
plt.plot(H_vec, color= 'black', label= 'Kalman filter')
plt.legend()
plt.show()


