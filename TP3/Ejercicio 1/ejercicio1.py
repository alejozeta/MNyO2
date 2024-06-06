import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
compression_values = [2, 6, 10]
j = 0
# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Ejercicio 1/dataset01.csv')

# Remove the first row (column names)
#df = df.iloc[1:]

# Remove the first column (row numbers)
df = df.iloc[:, 1:]

# Convert the remaining columns to float
df = df.astype(float)

# Convert the DataFrame to a numpy array
X = df.values
compression_values += [X.shape[1]]
print(X.shape[1])
Y = np.loadtxt('Ejercicio 1/y1.txt')



X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

U,S,VT = np.linalg.svd(X_centered, full_matrices=False)
print(U.shape, S.shape, VT.shape)

xyz = np.zeros((X.shape[0],3))
for i in range(X.shape[0]):
    xyz[i,0] = VT[0,:]@X_centered[i,:]
    xyz[i,1] = VT[1,:]@X_centered[i,:]
    xyz[i,2] = VT[2,:]@X_centered[i,:]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the x, y, and z coordinates
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

# Set labels for the x, y, and z axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.title('Clusters')

# Show the plot
plt.show()


def similarity_matrix(matrix, sigma):
    distances = squareform(pdist(matrix, 'euclidean'))
    k = np.exp(-distances**2/(2*sigma**2))
    return k

def PCA(U, S, d):
    Sd = np.diag(S[:d])
    Ud = U[:, :d]
    Z = np.dot(Ud, Sd)
    return Sd, Ud, Z

plt.figure(figsize=(8, 8))
counter = 1
for compression in compression_values:
    VT_hat = VT[:compression, :]
    # Perform PCA
    S_hat, U_hat, XV_hat = PCA(U, S, compression)

    


    sigma = 1.0
    
    # Calculate the similarity matrix
    similarity = similarity_matrix(XV_hat, sigma)
    
    # Display the similarity matrix
    plt.subplot(2,2,counter)
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar()
    plt.title('Compresión = ' + str(compression))
    # plt.show()
    counter+=1
plt.show()

        


doms = np.linspace(1,206,206)
#Singular values plot:
plt.figure(figsize=(8,6))
plt.bar(doms,S[:compression])
# plt.title('Singular values plot of the full matrix')
plt.ylabel('Valor singular')
plt.xlabel('Dimensión')
plt.show()