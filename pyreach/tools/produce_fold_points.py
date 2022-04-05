import numpy as np
import matplotlib.pyplot as plt

np_file = np.load('/home/kaushiks/pyreach-internal/pyreach/tools/shirt_script_8.npy', allow_pickle=True).item()
# print(np_file.item()['points'])

plt.imshow(np_file['image'])
plt.show()

pts = [(368, 2),
(200, 88.5), 
(516.2741935483871, 44.79032258064518), 
(273.6935483870968, 262.2096774193548), 
(218.01612903225808, 278.33870967741933),
(334.98387096774195, 222.85483870967744),
(228, 78),
(345, 246),]

np.save('/home/kaushiks/pyreach-internal/pyreach/tools/shirt_script_9.npy', {'image': np_file['image'], 'mask': np_file['mask'], 'points': np.array(pts)})