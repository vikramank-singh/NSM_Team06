import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TT = pd.read_csv("T.txt", header=None)
vv = pd.read_csv("u.txt", header=None)
II = pd.read_csv("I.txt", header=None)

nx = 201
ny = 201
nz = 20

T = np.zeros((nz,ny,nx))
v = np.zeros((nz,ny,nx))
I = np.zeros((nz,ny,nx))

T = TT.values.reshape((nz,ny,nx))
v = vv.values.reshape((nz,ny,nx))
I = II.values.reshape((nz,ny,nx))


fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
plt.pcolormesh(T[nz//2,:,:], cmap='coolwarm')
plt.colorbar()
plt.title("Temperature (K) profile at mid plane (z=Lz/2)")

plt.sca(ax[1])
plt.pcolormesh(v[nz//2,:,:], cmap='coolwarm')
plt.colorbar()
plt.title(" u profile at mid plane (z=Lz/2)")

plt.sca(ax[2])
plt.pcolormesh(I[-1,:,:], cmap='coolwarm')
plt.colorbar()
plt.title(" I profile at exit plane (z=Lz)")


fig, ax = plt.subplots(2,1)
plt.sca(ax[0])
plt.pcolormesh(np.rot90(I[:,:,nx//2]), cmap='coolwarm')
plt.colorbar()

plt.sca(ax[1])
plt.pcolormesh(np.rot90(I[:,nx//2,:]), cmap='coolwarm')
plt.colorbar()
plt.show()
