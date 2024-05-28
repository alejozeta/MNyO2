import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def similarity_matrix(X, sigma):
    distances = np.linalg.norm(X)
    k = np.exp(-distances**2/(2*sigma**2))
    return k


compression_values = [2, 6, 10, 206]
j = 0
# Cargar los datos
X = pd.read_csv('dataset01.csv').values 
Y = np.loadtxt('y1.txt')

U,S,VT = np.linalg.svd(X, full_matrices=False)
print(U.shape, S.shape, VT.shape)
Z = U@S
print(Z.shape)

for compression in compression_values:
    
    #Matrix U
    U_hat = U[:,:compression]
    plt.figure(figsize=(8, 6))
    plt.subplot(2,2,1)
    #Heatmap of matrixU
    sns.heatmap(U_hat, cmap='coolwarm', center=0)
    plt.title('Matrix U with compression = ' + str(compression))
    

    #Matrix S
    S_hat = np.diag(S[:compression])
    #Plot the matrix S
    # plt.figure(figsize=(8, 6))
    plt.subplot(2,2,2)
    plt.imshow(S_hat, cmap='coolwarm')
    plt.colorbar()
    plt.title('Matrix S with compression = ' + str(compression))
    
    plt.subplot(2,2,3)
    # VT matrix
    VT_hat = VT[:compression, :]
    # Heatmap of matrix VT
    sns.heatmap(VT_hat, cmap='coolwarm', center=0)
    plt.title('Matrix VT with compression = ' + str(compression))
    # plt.show()
    plt.subplot(2,2,4)
    #Matrix XV 
    XV_hat = U_hat @ S_hat 
    # Heatmap of matrix XV
    sns.heatmap(XV_hat, cmap='coolwarm', center=0)
    plt.title('Matrix XV with compression = ' + str(compression))
    plt.show()

    