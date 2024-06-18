import numpy as np
import matplotlib.pyplot as plt
n = 5
d = 100
b = np.random.randn(n)
x0 = np.random.randn(d)

iters = 1000

def F(matrix, x):
    return np.linalg.norm(matrix @ x - b)**2
    

def grad_f(matrix, x):
    return 2 * matrix.T @ (matrix @ x - b)

def hess_f(matrix):
    return 2 * matrix.T @ matrix

def F2(matrix, x,delta):
    return F(x) + delta * F(x)*np.dot(x,x)

def grad_f2(matrix, x,delta):
    return grad_f(matrix,x) + 2 * delta * F(matrix,x) * x

def calculate_alpha(matrix):
    """
    Alpha = 1/max(eigenvalues(Hessian))
    """
    return 1 / np.max(np.linalg.eigvals(hess_f(matrix)))

def gradiente_descendente(matrix, x0, iters, alpha):
    x = x0
    for i in range(iters):
        x = x - alpha * grad_f(matrix, x)
    return x

def main1():
    matrix = np.random.randn(n, d)
    alpha = calculate_alpha(matrix)
    x = gradiente_descendente(matrix, x0, iters, alpha)
    plt.figure()
    plt.plot(x)
    plt.show()



    

