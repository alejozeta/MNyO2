import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.spatial.distance import pdist, squareform

images = ['datasets_imgs/img00.jpeg', 'datasets_imgs/img01.jpeg', 'datasets_imgs/img02.jpeg', 'datasets_imgs/img03.jpeg', 
            'datasets_imgs/img04.jpeg', 'datasets_imgs/img05.jpeg', 'datasets_imgs/img06.jpeg', 'datasets_imgs/img07.jpeg', 
            'datasets_imgs/img08.jpeg', 'datasets_imgs/img09.jpeg', 'datasets_imgs/img10.jpeg', 'datasets_imgs/img11.jpeg', 
            'datasets_imgs/img12.jpeg', 'datasets_imgs/img13.jpeg', 'datasets_imgs/img14.jpeg', 'datasets_imgs/img15.jpeg', 
            'datasets_imgs/img16.jpeg', 'datasets_imgs/img17.jpeg', 'datasets_imgs/img18.jpeg']

# Mostrar imágenes originales
plt.figure(figsize=(10, 5))
for i in range(len(images)):
    plt.subplot(4, 5, i+1)
    plt.imshow(imread(images[i]), cmap='gray')
    plt.axis('off')
plt.suptitle('Original images')
plt.show()

# Valores de compresión
compression_values = [2, 6, 10]

images_data = []
for img in images:
    image_as_matrix = imread(img)
    images_data.append(image_as_matrix.flatten())

images_data = np.array(images_data)

# Descomposición SVD
U, S, Vt = np.linalg.svd(images_data, full_matrices=False)

for d in compression_values:
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    image_approximation = U_d @ S_d @ Vt_d

    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        plt.subplot(4, 5, i+1)
        plt.imshow(image_approximation[i].reshape(image_as_matrix.shape), cmap='gray')
        plt.axis('off')
    plt.suptitle('Compression = ' + str(d))
    plt.show()

# Valores singulares y suma acumulada
plt.figure(1)
plt.semilogy(np.diag(np.diag(S)))
plt.title('Singular values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(np.diag(S)))/np.sum(np.diag(np.diag(S))))
plt.title('Cumulative sum of singular values')
plt.show()

# Matriz de similaridad
for d in compression_values:
    U_hat = U[:, :d]
    S_hat = np.diag(S[:d])
    Vt_hat = Vt[:d, :]

    image_approximation = U_hat @ S_hat @ Vt_hat
    distances = pdist(image_approximation, 'euclidean')
    similarity_matrix = squareform(distances)

    plt.figure()
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Similarity Matrix for d = {d}')
    plt.show()

