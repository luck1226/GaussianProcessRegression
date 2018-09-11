# test for GPR slowfunction

# To run with command: python GP_Fourier.py 40 97 0.015

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from GP_Basic2 import *

# input array of numbers
# first input is # of train-point
# second is # of test-point

num = sys.argv

s, s0 = 1, 0.05
# number of training points
train_N = int(num[1])

# number of testing points
star_N = int(num[2])

# input lengthscale
ll = float(num[3])

# initial the test points randomly selected from [0,1]
x_star = np.linspace(0, 1, star_N)
x_function = np.linspace(0, 1, 100)

def f(x):
	#return np.exp((-1)/(x+0.00001))
	#return (x-1)**2
    #return 1.5*x**4 - x**2
    #return np.sin(2*np.pi/(x+0.001))
    return np.sin(12*np.pi*x)*np.cos(2*np.pi*x)

def kth_eigvec(k, N):
    theta = k*np.pi/(N+1)
    tmp = np.zeros((N,1))
    phi = (2*N+1.)*theta
    norm = (N+1.)/2.
    for i in range(N):
        tmp[i] = np.sin((i+1)*theta)
    return tmp/np.sqrt(norm)

def predict_y(N, M, y_train, k_star, epsilon, s, s0):
    y_predict = np.zeros((M,1))
    var = np.zeros((M,1))
    count = 0
    for i in range(N):
        tmp = np.zeros((M,1))
        vec = kth_eigvec(i+1, N)
        theta_k = (i+1)*np.pi/(N+1)
        flag_ = 1+2*epsilon*np.cos(theta_k)+2*epsilon**4*np.cos(2*theta_k)
        lam_da = s**2*flag_+s0**2
        tmp = np.dot(k_star, np.conjugate(vec))
        var += np.square(tmp) * (1/lam_da)
        tmp *= np.dot(np.matrix.transpose(vec),y_train)
        tmp *= (1/lam_da)
        y_predict = y_predict + tmp
    var = s**2*np.ones((M,1)) - var
    var = np.sqrt(var)
    return y_predict, var

def var(N, M, x_train, k_star, epsilon, s, s0):
    return 0

'''
begin of init parameters
'''


x_train = np.linspace(0, 1, train_N)

y_train = f(x_train)

#muu = y_train.mean()
#y_train -= muu
#np.random.seed(42)
#y_train = y_train + np.random.normal(0, 0.05 ,len(y_train))

y_train = np.matrix.transpose(y_train)

epsilon = np.exp((-0.5) / np.square((train_N-1)*ll))

k_star = covK_star(x_star, x_train, ll, s)

y_star, var = predict_y(train_N, star_N, y_train, k_star, epsilon, s, s0)
#y_star += muu


#ll1 = ll*10
#epsilon1 = np.exp((-0.5) / np.square((train_N-1)*ll1))
#k_star1 = covK_star(x_star, x_train, ll1, s)
#y_star1 = predict_y(train_N, star_N, y_train, k_star1, epsilon1, s, s0)
#y_star1 += muu

y_function = f(x_star)
#y_err = y_function - y_star
#y_err = np.square(y_err)
#print sum(y_err[:,0])/len(y_err)

#f_full_res = []
#filename = 'fullGP004.csv'
#with open(filename) as csvfile:
#    reader = csv.reader(csvfile, delimiter=',')
#    for row in reader:
#        f_full_res.append(row[1])

f = plt.figure()
#plt.subplot(211)
plt.plot(x_star, y_function, 'k-.')
plt.plot(x_train, y_train, 'ro')
plt.plot(x_star, y_star, 'b-')
#plt.plot(x_star, f_full_res,'g*')
#plt.plot(x_star, y_star+var, 'r-')
#plt.title('Grid w/ 2nd n.n. correlation ( length scale = {})'.format(ll))
#plt.xticks([])

#plt.subplot(212)
#plt.plot(x_function, y_function, 'y--')
#plt.plot(x_train, y_train+muu, 'r.')
#plt.plot(x_star, y_star1, 'b-')
#plt.title('Length scale = {}'.format(ll1))
plt.show()
#f.savefig("regression_re2.pdf", bbox_inches='tight')
