import matplotlib.pyplot as plt
import numpy as np

# plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.plot(x, y)

plt.show()
