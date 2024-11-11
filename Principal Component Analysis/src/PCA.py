import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
import sklearn.decomposition as dc
import pickle
from sklearn.utils import Bunch

load_pickled_data = True

if load_pickled_data:
    with open('faces.pickle', 'rb') as handle:
        faces = pickle.load(handle)
else:
    # This .pickle object was created by the following code: 
    # (you can use it to load the data from sklearn yourself, but download seems slow in noto)
    from sklearn.datasets import fetch_lfw_people
    min_faces_per_person = 30
    faces = fetch_lfw_people(min_faces_per_person=min_faces_per_person)

with open('faces.pickle', 'wb') as handle:
    pickle.dump(faces, handle)

# explore a bit shape, dtype, ...
print(faces.keys())
print(np.shape(faces.images))

images = faces["images"][:10]
fig,ax = plt.subplots(2,5, figsize=(10,5))
ax=ax.ravel()

for i, (img,ax) in enumerate(zip(images,ax)):
    ax.imshow(img,cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Keep only 30 samples per person.
indices = np.hstack([np.where(faces["target"] == target)[0][:30] for target in np.unique(faces["target"])])
faces["images"] = faces["images"][indices]
faces["target"] = faces["target"][indices]

unique, counts = np.unique(faces.target, return_counts=True)
pca = dc.PCA().fit(faces.data)

# Plot the first 5 principal components. Hint: need to reshape the vectors to 2d images.
a = pca.components_.reshape((2041, 62, 47))
a = a[:5]
fig,ax = plt.subplots(1,5, figsize=(10,5))
ax=ax.ravel()

for i, (img,ax) in enumerate(zip(a, ax)):
    ax.imshow(img,cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Plot the cumulative explained variance of the principal compoments
plt.plot(range(0,len(pca.explained_variance_)), pca.explained_variance_ratio_.cumsum()) # todo
plt.xlabel("pca component")
plt.ylabel("cumulative explained variance")

# Implement a function which makes the plot given a number of components, a PCA object, and the data samples. 
def show_projected_images(n_components, pca_object, images):
    nsamples=images.shape[0]
    tf = pca_object.transform(images)
    tf[:, n_components:]=0
    tf_inv = pca_object.inverse_transform(tf).reshape((nsamples,62, 47))
    fig, axes = plt.subplots(1, 4)
    for i, ax in enumerate(axes.flat):
        ax.imshow(tf_inv[i], cmap='gray')
    pass # todo

# use the show_projected_images() function 
components_list = [10, 30, 100, 300, 1000]

for n in components_list:
    show_projected_images(n, pca, faces.data)
