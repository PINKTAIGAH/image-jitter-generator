import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, sig)
# ax2.imshow(abs(cwtmatr))
ax2.plot(t, abs(cwtmatr[4]))
plt.show()
