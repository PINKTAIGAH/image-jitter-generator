import torch
import config
import numpy as np
import matplotlib.pyplot as plt
from ImageGenerator import ImageGenerator
from torchvision.transforms import Pad

filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE)
transform = Pad((20, 20))

x,_ = filter.generateGroundTruth()
x = transform(x)
y = torch.clone(x)
y_1 = torch.clone(x)
z = torch.clone(x)
A = 5
F = 20
PHI = 2

for i in range(config.IMAGE_SIZE):
    # shift = A * np.sin(20*np.pi*i/276) * np.sin(5*np.pi*i/276)
    shift = A * np.sin(15*2*np.pi*i/276 + PHI)
    y[:, i] = torch.roll(x[:, i], shifts=int(shift))

for i in range(config.IMAGE_SIZE):
    # shift = A * np.sin(20*np.pi*i/276) * np.sin(5*np.pi*i/276)
    shift = A * np.sin(F*2*np.pi*i/276 + PHI)
    z[i, :] = torch.roll(y[i, :], shifts=int(shift))

for i in range(config.IMAGE_SIZE):
    # shift = A * np.sin(20*np.pi*i/276) * np.sin(5*np.pi*i/276)
    shift = A * np.sin(F*2*np.pi*i/276 + PHI)
    y_1[i, :] = torch.roll(x[i, :], shifts=int(shift))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(x)
ax2.imshow(y_1)
ax3.imshow(y)
plt.show()
