import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.morlet, widths)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, sig)
ax2.plot(t, cwtmatr[0])
plt.show()
