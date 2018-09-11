
# Developer: Chi-Ken Lu
# First version: Feb 26th 2018
# Sponsored by CoDaS Lab

# GP Standing Wave Decomposition applying to two-dimensional input space
# Using squared exponential Kernel, we may exploit the matrix structure
# of Kronecker product

import numpy as np
import matplotlib.pyplot as plt
from GP_Basic1 import *

Nx = 10
Ny = Nx
Tx = 53
ll = 0.06
Delta = 1./(Nx-1)
alpha = np.exp((-0.5)*np.square(Delta/ll))

ll1 = ll/2
Delta1 = 1./(2*Nx-1)
alpha1 = np.exp((-0.5)*np.square(Delta1/ll1))

s = 1.2
s0 = 0.0001

def function_x(x):
    return np.sin(4*np.pi*x)

def function_y(y):
    return np.cos(4*np.pi*y)
    #return y

def train_output(train_x, train_y):
    train_f = np.zeros((len(train_x)*len(train_y),1))
    k = 0
    for i in range(len(train_y)):
        for j in range(len(train_x)):
            input_x = train_x[j]
            input_y = train_y[i]
            val = function_x(input_x)*function_y(input_y)
            train_f[k] = val
            k = k +1
    return train_f

def kth_eigvec(k, N):
    theta = k*np.pi/(N+1)
    tmp = np.zeros((N,1))
    phi = (2*N+1.)*theta
    norm = (N+1.)/2.
    for i in range(N):
        tmp[i] = np.sin((i+1)*theta)
    return tmp/np.sqrt(norm)

def inv_Kernel_UU(train_x, train_y, alpha, s, s0):
    Nx = len(train_x)
    Ny = len(train_y)
    inv_K_UU_x = np.zeros((Nx, Nx))
    inv_K_UU_y = np.zeros((Ny, Ny))

    for i in range(Nx):
        vec = kth_eigvec(i+1,Nx)
        theta_i = (i+1)*np.pi/(Nx+1)
        eig_val_fac = 1 + 2 * alpha * np.cos(theta_i)
        eig_val = s**2 * eig_val_fac + s0**2
        inv_K_UU_x = inv_K_UU_x + eig_val**(-1) * np.dot(vec,np.matrix.transpose(vec))

    inv_K_UU_y = inv_K_UU_x
    return np.kron(inv_K_UU_y, inv_K_UU_x)

def cross_corr_XU(test_x, test_y, train_x, train_y, ll, s):
    K_x = covK_star(test_x, train_x, ll, s)
    K_y = covK_star(test_y, train_y, ll, s)
    return np.kron(K_y, K_x)


#test_x = 2./np.pi
#test_x = np.array([test_x])

#test_y = 0.5/np.pi
#test_y = np.array([test_y])

#print "Answer is {}".format(function_x(test_x)*function_y(test_y))

train_x = np.linspace(0,1,Nx)
train_x1 = np.linspace(0,1,2*Nx)

train_y = np.linspace(0,1,Ny)
train_y1 = np.linspace(0,1,2*Ny)

train_f = train_output(train_x, train_y)
train_f1 = train_output(train_x1, train_y1)

#cross_test_U = cross_corr_XU(test_x, test_y, train_x, train_y, ll, s)

inv_K = inv_Kernel_UU(train_x, train_y, alpha, s, s0)
inv_K1 = inv_Kernel_UU(train_x1, train_y1, alpha1, s, s0)
#pred = np.dot(cross_test_U, np.dot(inv_K, train_f))

#print "Prediction is {}".format(pred)

H1 = []
H2 = []
H3 = []

t_x = np.linspace(0,1,Tx)
t_y = np.linspace(0,1,Tx)

for yy in t_y:
    val_x = []
    for xx in t_x:
        val_x.append(function_x(xx)*function_y(yy))
    H1.append(val_x)


for ty in t_y:
    val_t = []
    val_t1 = []
    for tx in t_x:
        testxx = np.array([tx])
        testyy = np.array([ty])
        cross_test_U = cross_corr_XU(testxx, testyy, train_x, train_y, ll, s)
        cross1 = cross_corr_XU(testxx, testyy, train_x1, train_y1, ll1, s)
        pred_t = np.dot(cross_test_U, np.dot(inv_K, train_f))
        pred_t1 = np.dot(cross1, np.dot(inv_K1, train_f1))
        val_t.append(pred_t[0][0])
        val_t1.append(pred_t1[0][0])
    H2.append(val_t)
    H3.append(val_t1)

#fig = plt.figure(figsize=(6, 3.2))

#ax = fig.add_subplot(111)
#ax.set_title('colorMap')
f = plt.figure()
plt.subplot(131)
plt.contour(t_y, t_x, H1, 6)
#plt.title('True function')
plt.ylabel('Y',fontsize=14)
#plt.clabel(CS, fontsize=9, inline=1)
#ax.set_aspect('equal')

plt.subplot(132)
plt.contour(t_y, t_x, H2, 6)
#plt.title('10x10 grid')
plt.xlabel('X',fontsize=14)
plt.yticks([])
#plt.set_yticklabels([])

plt.subplot(133)
plt.contour(t_y, t_x, H3, 6)
#plt.title('20x20 grid')
plt.yticks([])
#plt.set_yticklabels([])
#plt.clabel(CS1, fontsize=9, inline=1)
#cax = fig.add_axes([0, 1, 0, 1])
#cax.get_xaxis().set_visible(False)
#cax.get_yaxis().set_visible(False)
#cax.patch.set_alpha(0)
#cax.set_frame_on(False)
#plt.colorbar(orientation='vertical')
plt.show()
f.savefig("2Dregression.pdf", bbox_inches='tight')
