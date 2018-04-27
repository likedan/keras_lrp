import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

image = np.load("Images-neckline_design_labels-0cb381e52f4b738a05fc55ec5fa76779.jpg.npy")
value = image+0.00000001
plt.imshow(np.log(np.log(np.abs(image))), cmap='jet')
print(value)
plt.show()