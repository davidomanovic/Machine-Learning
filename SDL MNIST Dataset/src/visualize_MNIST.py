import numpy as np
import matplotlib.pyplot as plt

class visualize_MNIST:
    def __init__(self):
      pass

    def plotimgs(imgs, nplot=4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    
    Parameters
    -----------
    imgs: (height*width,N) array containing images, where N > nplot**2
    nplot: integer, nplot**2 images will be plotted

    """

    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    assert (n > nplot**2), "Need amount of data in matrix N > nplot**2"

    # Initialize subplots
    fig, axes = plt.subplots(nplot, nplot, figsize=(12, 9))

    # Set background color
    plt.gcf().set_facecolor("lightgray")

    # Iterate over images
    for idx in range(nplot**2):

        # Break if we go out of bounds of the array
        if idx >= n:
            break

        # Indices
        i = idx//nplot
        j = idx % nplot

        # Remove axis
        axes[i, j].axis('off')

        axes[i, j].imshow(imgs[:, idx].reshape((m, m)), cmap="gray")

    # Plot
    fig.tight_layout()
    plt.show()

