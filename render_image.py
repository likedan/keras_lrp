import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

image = np.load("Images-neckline_design_labels-6e7c447b97b5017fb3f76665fd1ced4e.jpg.npy")
value = image+0.00000001
reduced_image = image.sum(axis=2)
print(reduced_image)
print(image)
print(reduced_image.shape)
plt.imshow(np.log(np.abs(image[:,:,2])), cmap='Greys')
plt.show()