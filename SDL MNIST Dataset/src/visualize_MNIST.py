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

    def plotOrthProj(b, D, b_tilde):
        """
        Takes in data from an arbitrary image, and applies the SVD approach on the data.
        Then it projects both data sets on the basis made from the A matrix, and plots the
        result som both projections for different values of d
    
        Parameters
        -----------
        A: 2d array with images from the given data, on the from of date[:, int, :]
        D: an array of values d: factor to remove the (m-d) last columns of the SVD
        bcompare: 2d array with images from the given data, on the from of date[:, int, :] with a diffrent int than A
    
        """
        plt.figure(figsize=(10, 7)) 
    
        # Sets subplots
        fig, axes = plt.subplots(len(D) + 1, 2, figsize=(12,9))
    
        # Calculate the SVD
        U, Σ, Vh = np.linalg.svd(b, full_matrices=False)
        Σ = np.diag(Σ)
    
        # For loop for each value of d in D
        for i in range(len(D)):
    
            # Gets the basis for A with the given value of d in D in
            Σ_d, W, H = truncSVD(U, Σ, Vh, D[i])
    
            # Calculates the projection of A and bcompare to the basis of A
            P = orthproj(W, b)
            Pb = orthproj(W, b_tilde)
    
            # Plots each of the images
            axes[i, 0].axis('off')
            axes[i, 0].imshow(P[0].reshape((28, 28)), cmap='gray')
            axes[i, 0].title.set_text(r"$P_{W}(b)$, d=" + str(D[i]))
            axes[i, 1].axis('off')
            axes[i, 1].imshow(Pb[0].reshape((28, 28)), cmap='gray')
            axes[i, 1].title.set_text(r"$P_{W}(\tilde{b})$, d=" + str(D[i]))
    
        fig.tight_layout()
        axes[len(D), 0].axis('off')
        axes[len(D), 0].imshow(P[0].reshape((28, 28)), cmap='gray')
        axes[len(D), 0].title.set_text("Original b")
        axes[len(D), 1].axis('off')
        axes[len(D), 1].imshow(Pb[0].reshape((28, 28)), cmap='gray')
        axes[len(D), 1].title.set_text(r"Original $\tilde{b}$")
    
        plt.show()

    def distances_SVD(A, D, bcompare, onlyComp=False):
        """
        Takes in data in the form of images, and applies the SVD approach on the data
        and calculates the distances of the images to the basis of the images
        it is trained to.
    
        Parameters
        -----------
        A: 2d array with images from the given data, on the from of date[:, int, :]
        D: an array of values d: factor to remove the (m-d) last columns of the SVD
        bcompare: 2d array with images from the given data, on the from of date[:, int, :] with a diffrent int than A
        plot: bool that decides if the functions should be plotted.
        onlyComp: a bool that decides if it only calculates the norm for the compear function, it is used in Task 3
    
        Returns
        -----------
        norms: an array of the norm values for the matrix A to its basis
        normsb: an array of the norm values for the matrix B to the basis of A basis
    
        """
        # Calculate the SVD
        U, Σ, Vh = np.linalg.svd(A, full_matrices=False)
        Σ = np.diag(Σ)
    
        # Test if onlyComp is set to True, this is used in Task 3
        if onlyComp == True:
    
            # Gets the basis for A with the given value of D
            W = truncSVD(U, Σ, Vh, D)[1]
    
            # Calculates the distance between the projection of b onto W and b
            distances = columndistance(orthproj(W, bcompare), bcompare)
    
            # Returns distances
            return distances
    
        # Runs if onlyComp is set to False, is used for this task
        else:
            # Set empty arrays to be used for each of the values of d
            distancesA = np.zeros(len(D))
            distances = np.zeros(len(D))
    
            # For loop for each value of d in D
            for (i, d) in enumerate(D):
                # Gets Σ_d, W, H from the function truncSVD
                Σ_d, W, H = truncSVD(U, Σ, Vh, d)
                # Calculates the norm of the difference between the projection of A onto W and A
                distancesA[i] = (np.linalg.norm(A - orthproj(W, A)))**2
                # Calculates the norm of the difference between the projection of b onto W and b
                distances[i] = (np.linalg.norm(bcompare - orthproj(W, bcompare)))**2
    
            # Plots the distance values for A and bcompare for each d value.
            plt.figure(figsize=(12, 9))
            plt.title(rf"$||A-P_W(A)||_F^2$ norm with. $d$ basis vectors", fontsize=12)
            plt.semilogy(D, distancesA, label="$||A-P_W(A)||_F^2$")
            plt.semilogy(D, distances, label="$||b-P_W(b)||_F^2$")
            plt.xlabel(rf"$d$ amount of basis vectors", fontsize=12)
            plt.ylabel(rf"Value (log scale)", fontsize=12)
            plt.legend()
            plt.show()
    
            # Returns the distances corresponding to A and bcompare
            return distancesA, distances
    

