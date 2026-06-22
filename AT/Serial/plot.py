import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

II = pd.read_csv("I.txt", header=None)
TT = pd.read_csv("T.txt", header=None)

nx = 121
ny = 121
nz = 20

I = np.zeros((nz,ny,nx))
T = 0.0*I
I = II.values.reshape((nz,ny,nx))
T = TT.values.reshape((nz,ny,nx))




fig, ax = plt.subplots(1,3)

plt.sca(ax[0])
plt.pcolormesh(I[0,:,:], cmap='coolwarm')
plt.colorbar()
plt.title("Laser profile at inlet (z=0)")

plt.sca(ax[1])
plt.pcolormesh(I[-1,:,:], cmap='coolwarm')
plt.colorbar()
plt.title("Laser profile at exit (z=Lz)")

plt.sca(ax[2])
plt.pcolormesh(T[nz//2,:,:], cmap='coolwarm')
plt.colorbar()
plt.title("Temperature (K) profile at mid plane (z=Lz/2)")


fig, ax = plt.subplots(2,1)
plt.sca(ax[0])
plt.pcolormesh(np.rot90(I[:,:,nx//2]), cmap='coolwarm')
plt.colorbar()

plt.sca(ax[1])
plt.pcolormesh(np.rot90(I[:,nx//2,:]), cmap='coolwarm')
plt.colorbar()
plt.show()
