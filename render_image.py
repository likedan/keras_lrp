import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

image = np.load("Images-neckline_design_labels-3d784813cda050811eb25b7096e722ce.jpg.npy")
value = image+0.00000001
plt.imshow(np.log(np.log(np.abs(image))))
print(value)
plt.show()