import numpy as np
import matplotlib.pyplot as plt

H = np.array([[16, 2, 3, 4],
			  [5, 6, 7, 8],
			  [9, 10, 11, 12],
			  [13, 14, 15, 16]])  # added some commas and array creation code

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.2, 0.1, 0.78, 0.8])

cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(1)
cax.set_frame_on(False)

plt.colorbar(orientation='vertical')
plt.show()
ax.grid(True)