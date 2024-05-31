import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

imagenes = ['datasets_imgs/img00.jpeg', 'datasets_imgs/img01.jpeg', 'datasets_imgs/img02.jpeg', 'datasets_imgs/img03.jpeg', 
            'datasets_imgs/img04.jpeg', 'datasets_imgs/img05.jpeg', 'datasets_imgs/img06.jpeg', 'datasets_imgs/img07.jpeg', 
            'datasets_imgs/img08.jpeg', 'datasets_imgs/img09.jpeg', 'datasets_imgs/img10.jpeg', 'datasets_imgs/img11.jpeg', 
            'datasets_imgs/img12.jpeg', 'datasets_imgs/img13.jpeg', 'datasets_imgs/img14.jpeg', 'datasets_imgs/img15.jpeg', 
            'datasets_imgs/img16.jpeg', 'datasets_imgs/img17.jpeg', 'datasets_imgs/img18.jpeg']

for img in imagenes:
    matrix_as_image = imread(img)

    image = plt.imshow(matrix_as_image)
    image.set_cmap('gray')
    plt.axis('off')
    plt.show()

    U, S, Vt = np.linalg.svd(matrix_as_image, full_matrices=False)
    S = np.diag(S)

    compression_values = [2, 6, 10]
    j = 0
    for r in compression_values:
        image_approximation = U[:, :r] @ S[0:r, :r] @ Vt[:r, :]
        plt.figure(j+1)
        j += 1
        image = plt.imshow(image_approximation)
        image.set_cmap('gray')
        plt.axis('off')
        plt.title('r = ' + str(r))
        plt.show()

    plt.figure(1)
    plt.semilogy(np.diag(S))
    plt.title('Singular values')
    plt.show()

    plt.figure(2)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    plt.title('Cumulative sum of singular values')
    plt.show()