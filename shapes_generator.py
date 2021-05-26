import matplotlib.pyplot as plt
from skimage.draw import random_shapes
import shutil
import os

# Delete old datasets.
try:
    shutil.rmtree('data')
except:
    pass

# Create new datasets.
for folder in ['test', 'train', 'val']:
    path = os.path.join('data', folder)
    for subfolder in ['circle', 'rectangle', 'triangle']:
        subpath = os.path.join(path, subfolder)
        os.makedirs(subpath)

# training shapes
for i in range(500):

    result = random_shapes((224, 224), shape='triangle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/train/triangle/%d.png' % i, image)

    result = random_shapes((224, 224), shape='circle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/train/circle/%d.png' % i, image)

    result = random_shapes((224, 224), shape='rectangle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/train/rectangle/%d.png' % i, image)

# val shapes
for i in range(100):

    result = random_shapes((224, 224), shape='triangle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/val/triangle/%d.png' % i, image)

    result = random_shapes((224, 224), shape='circle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/val/circle/%d.png' % i, image)

    result = random_shapes((224, 224), shape='rectangle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/val/rectangle/%d.png' % i, image)

# test shapes
for i in range(100):

    result = random_shapes((224, 224), shape='triangle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/test/triangle/%d.png' % i, image)

    result = random_shapes((224, 224), shape='circle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/test/circle/%d.png' % i, image)

    result = random_shapes((224, 224), shape='rectangle',
                           max_shapes=1, min_size=20)
    image, labels = result
    plt.imsave('data/test/rectangle/%d.png' % i, image)
