import config
import numpy as np
import torch
from ImageGenerator import ImageGenerator
import matplotlib.pyplot as plt
from torchvision.transforms import Pad

filter = ImageGenerator(config.PSF, config.MAX_JITTER, config.IMAGE_SIZE)
transform = Pad((2, 2))

x, _ = filter.generateGroundTruth()
ft_x = torch.fft.fft2(x)
ift_x = torch.fft.ifft2(np.exp(-1j*2*np.pi*1)*ft_x)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# plt.imshow(torch.real(ft_x_shifted))
ax1.imshow(x)
ax2.imshow(torch.real(ift_x))
ax3.imshow(x - torch.real(ift_x))
plt.show()
