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

# 2.4
images_2 = ['datasets_imgs_02/img00.jpeg', 'datasets_imgs_02/img01.jpeg', 'datasets_imgs_02/img02.jpeg', 'datasets_imgs_02/img03.jpeg', 
            'datasets_imgs_02/img04.jpeg', 'datasets_imgs_02/img05.jpeg', 'datasets_imgs_02/img06.jpeg', 'datasets_imgs_02/img07.jpeg']

def frobenius_norm(matrix):
    return np.linalg.norm(matrix, 'fro')

images_dataset2 = []
for img in images_2:
    image_as_matrix = imread(img)
    images_dataset2.append(image_as_matrix.flatten())

images_dataset2 = np.array(images_dataset2)

# Descomposición SVD
U_dtst2, S_dtst2, Vt_dtst2 = np.linalg.svd(images_dataset2, full_matrices=False)

for d in range(1, 8):
    U_d_dtst2 = U_dtst2[:, :d]
    S_d_dtst2 = np.diag(S_dtst2[:d])
    Vt_d_dtst2 = Vt_dtst2[:d, :]
    
    image_approximation = U_d_dtst2 @ S_d_dtst2 @ Vt_d_dtst2

    frobenius = frobenius_norm(images_dataset2 - image_approximation)
    print(f'Error para d = {d}: {frobenius * 100}%')
    if frobenius < 0.1:
        print(f'El número mínimo de dimensiones que genera menos de 10% de error en la reducción es {d}')
        print(f'Error: {frobenius * 100}%')
        break
    
    # elif d == 7:
    #     print(f'No se encontró un número de dimensiones que genere menos de 10% de error en la reducción.  {frobenius * 100}%')