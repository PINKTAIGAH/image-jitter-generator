from JitterDataset import JitteredDataset
import matplotlib.pyplot as plt

fig , (ax1, ax2) = plt.subplots(1,2)
axes = (ax1, ax2)
dataset = JitteredDataset(256, 4, 5, 5)

truth, jittered = dataset[0][0], dataset[0][1]

ax1.imshow(truth, cmap="gray")
ax2.imshow(jittered, cmap="gray")
plt.show()



