# running with command: python GP_random2.py 20 97 7 0.03
# first number --- number of grid points
# second number --- number of test points
# third number --- data of random numper generated per grid point
# fourth number --- length scale (usually 0.6 times the grid spacing)

import numpy as np
import matplotlib.pyplot as plt
import sys
#from random import *

# input array of numbers
# first input is # of train-point
# second is # of test-point

num = sys.argv

s, s0 = 1., 0.05
# number of training points
train_N = int(num[1])

# number of testing points
star_N = int(num[2])

N_times = int(num[3])

# input lengthscale
ll = float(num[4])

# initial the test points randomly selected from [0,1]
x_star = np.linspace(-0.2, 1.2, star_N)
x_function = np.linspace(0, 1, 100)

def f(x):
	#return np.exp((-1)/(x+0.00001))
	#return (x-1)**2
    #return 1.5*x**4 - x**2
    #return np.sin(6*np.pi*(x+0.1))/(x+0.1)
    return x*np.cos(8*np.pi*(x+0.15))*np.cos(2*np.pi*x)

def cov_fn(x1, x2, ll, s):
    return s**2 * np.exp((-0.5)*np.square((x1-x2)/ll))

def covK(x_train, ll, s):
    N = x_train.shape[0]
    H = np.zeros((N,N))
    for k in range(len(x_train)):
        H[k,:] = cov_fn(x_train[k], x_train, ll, s)
    return H

def covK_star1(x_star, x_train, ll, s):
    H = np.zeros((len(x_star), len(x_train)))
    i = 0
    for k in x_star:
        H[i, :] = cov_fn(k, x_train, ll, s)
        i = i +1
    return H

def covK_star(x_star, x_train, ll, s):
    H = np.zeros((len(x_star), len(x_train)))
    for k in range(len(x_star)):
        bag = neighbor_index(x_star[k],x_train)
        for j in bag:
            H[k,j] = cov_fn(x_star[k], x_train[j], ll, s)
    return H

# The following function return 3 nearest points in training set to the test point
# When writing down the K_*, the covariance between test and training points
# The matrix elements are nonzero only for three places at most
#

def neighbor_index(x, x_train):
    bag = []
    if x <= min(x_train):
        bag.append(0)
        bag.append(1)
        return bag
    if x >= max(x_train):
        bag.append(len(x_train)-2)
        bag.append(len(x_train)-1)
        return bag
    low = 0
    high = len(x_train)-1
    mid = (low+high)//2
    while x<x_train[mid] or x>x_train[mid+1]:
        if x<x_train[mid]:
            high = mid
        else:
            low = mid
        mid = (low+high)//2
    if abs(x-x_train[mid]) > abs(x-x_train[mid+1]):
        bag.append(mid)
        bag.append(mid+1)
        if mid+2 < len(x_train):
            bag.append(mid+2)
    else:
        if mid-1 >= 0:
            bag.append(mid-1)
        bag.append(mid)
        bag.append(mid+1)
    return bag


def kth_eigvec(k, N):
    theta = k*np.pi/(N+1)
    tmp = np.zeros((N,1))
    phi = (2*N+1.)*theta
    norm = (N+1)/2
    for i in range(N):
        tmp[i] = np.sin((i+1)*theta)
    return tmp/np.sqrt(norm)

def predict_y(N, M, y_train, k_star, epsilon, s, s0):
    y_predict = np.zeros((M,1))
    count = 0
    for i in range(N):
        tmp = np.zeros((M,1))
        vec = kth_eigvec(i+1, N)
        theta_k = (i+1)*np.pi/(N+1)
        flag_ = 1+2*epsilon*np.cos(theta_k)
        lam_da = s**2*flag_+s0**2
        tmp = np.dot(np.matrix.transpose(vec),y_train) * np.dot(k_star, vec) * (1/lam_da)
        y_predict = y_predict + tmp
    return y_predict

def SWD_GP(X, Y, N1, k_star, X_u, epsilon, s, s0):

    N = X.shape[0]
    M = X_u.shape[0]
    K_xu = covK_star(X, X_u, ll, s)
    K_ux = np.matrix.transpose(K_xu)
    UU = []

    for k in range(M):
        UU.append(kth_eigvec(k+1,M))

    y_bar = np.zeros((N,1))
    inv_s_star = np.zeros((N,N))

    for i in range(N):
        ss_star = 0.
        xx= []
        xx.append(X[i])
        star_xu = covK_star(xx, X_u, ll, s)
        for j in range(M):
            theta = (j+1)*np.pi/(M+1)
            flag = 1+2.*epsilon*np.cos(theta)
            pref_ = 1/(s**2*flag)
            tmp_ = np.square(np.dot(star_xu,UU[j]))
            ss_star += pref_*tmp_
        ss_star = s**2 - ss_star + s0**2
        inv_s_star[i][i] = 1/ss_star

    gg = np.zeros((M,M))
    gg = np.dot(K_ux,np.dot(inv_s_star, K_xu))

    u_bar1 = np.zeros((M,1))
    u_bar1 = np.dot(K_ux, np.dot(inv_s_star, Y))

    u_bar = np.zeros((M,1))


    diagonal_sum = 0

    subdiagonal_sum = 0

    superdiagonal_sum = 0

    offdiagonal_sum = 0

    for i in range(M):

        diagonal_sum += gg[i][i]

        if i > 0:
            subdiagonal_sum += gg[i-1][i]
        if i < M-1:
            superdiagonal_sum += gg[i][i+1]

    diagonal_sum /= M
    subdiagonal_sum /= (M-1)
    superdiagonal_sum /= (M-1)
    offdiagonal_sum = 0.5 * (subdiagonal_sum + superdiagonal_sum)

    for i in range(M):
        gg[i][i] -= diagonal_sum
        if i > 0:
            gg[i-1][i] -= offdiagonal_sum
        if i < M-1:
            gg[i][i+1] -= offdiagonal_sum


    pred = np.zeros((N1,1))

    for i in range(M):

        theta = (i+1)*np.pi/(M+1)
        cor = np.dot(np.matrix.transpose(UU[i]),np.dot(gg, UU[i]))

        eig_val_i = (1+diagonal_sum) + 2*(epsilon+offdiagonal_sum)*np.cos(theta)
        lam_da_numerator = s**2*eig_val_i + s0**2
        lam_da_denominator = lam_da_numerator + cor
        tmp = (1 / lam_da_denominator) * np.dot(np.matrix.transpose(UU[i]),u_bar1) * np.dot(k_star, UU[i])
        pred += tmp

    return pred


'''

begin of init parameters

'''
x_train = np.linspace(0, 1, train_N)

np.random.seed(42)
X = np.random.uniform(0, 1, N_times*train_N)
Y = f(X) + np.random.normal(0, 0.1, len(X))

y_train = f(x_train)
muu = y_train.mean()
y_train -= muu

y_train = np.matrix.transpose(y_train)

epsilon = np.exp((-0.5) / np.square((train_N-1)*ll))

k_star = covK_star(x_star, x_train, ll, s)

y_star = SWD_GP(X, Y, star_N, k_star, x_train, epsilon, s, s0)

y_star += muu

y_function = f(x_function)

y_true = f(x_star)
y_err = y_true - y_star
y_err = np.square(y_err)
#print y_err.shape[0]
y_error = sum(y_err[:,0])/len(y_err)
#print y_error

f = plt.figure()
plt.plot(X,Y,'c.')
plt.plot(x_function, y_function, 'r--')
#plt.plot(x_train, y_train, 'r.')
plt.plot(x_star, y_star, 'b-')
#plt.plot(x_train, u_b, 'ro')
#plt.plot(x_train, u_bar, 'g+')
plt.ylim([-1.25, 1.25])
plt.text(-0.2,1,'(c)',fontsize=20)
plt.xlabel('X',fontsize=14)
plt.show()
f.savefig("datafitting_1stc.pdf", bbox_inches='tight')
