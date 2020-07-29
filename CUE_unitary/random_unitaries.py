from scipy import linalg
import numpy as np
from numpy import cos, sin, exp, pi
import matplotlib.pyplot as plt

def haar_measure(n):
    # A Random matrix distributed with Haar measure
    # taken from https://arxiv.org/pdf/math-ph/0609050.pdf
    z = (randn(n,n) + 1j*randn(n,n))/sqrt(2.0)
    q,r = linalg.qr(z)
    d = diagonal(r)
    ph = d/absolute(d)
    q = multiply(q,ph,q)

    return q

def u3(theta, phi, lam):
    # definition of u3 gate can be found here
    # https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html#qiskit.circuit.library.U3Gate
    return np.array([[cos(theta/2), -exp(1j*lam) * sin(theta/2)], \
                     [exp(1j*phi) * sin(theta/2), exp(1j*(phi+lam) * cos(theta/2))]])

thetas = []
spacings = []

thetas2 = []
spacings2 = []

for i in np.arange(100000):
    # method 1 (correct method from paper)
    u = haar_measure(2)
    ev, vecs = linalg.eig(u)

    t = np.sort(np.angle(ev))
    thetas.append(t[0])
    thetas.append(t[1])

    spacings.append( 1/np.pi * (t[1] - t[0]))

    rand = np.random.rand

    u = u3(rand()*pi, rand()*2*pi, rand()*2*pi)
    ev, vecs = linalg.eig(u)

    t = np.sort(np.angle(ev))
    thetas2.append(t[0])
    thetas2.append(t[1])

    spacings2.append( 1/np.pi * (t[1] - t[0]))
#%%
plt.figure()
plt.hist(thetas, bins=200, histtype='step', label='Method from paper', normed=True)
plt.hist(thetas2, bins=200, histtype='step', label='Current method', normed=True)
plt.legend()
plt.xlabel('$\\theta$')
plt.ylabel('$\\rho(\\theta)$')

plt.figure()
plt.hist(spacings, bins=200, histtype='step', label='Method from paper', normed=True)
plt.hist(spacings2, bins=200, histtype='step', label='Current method',normed=True)
plt.xlabel('$s$')
plt.ylabel('$\\rho(s)$')
plt.xlim((0,3))
plt.legend()
