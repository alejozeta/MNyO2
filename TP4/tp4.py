import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)


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

def F2(matrix, x, delta):
    return F(matrix,x) + delta * F(matrix,x)*np.dot(x,x)

def grad_f2(matrix, x,delta):
    return grad_f(matrix,x) + 2 * delta * F(matrix,x) * x

def hess_f2(matrix,delta):
    return hess_f(matrix) + 2 * delta * np.eye(matrix.shape[1])

def calculate_alpha(matrix):
    """
    Alpha = 1/max(eigenvalues(Hessian))
    """
    return 1 / np.max(np.linalg.eigvals(hess_f(matrix)))

def gradiente_descendente(matrix, x0, iters, alpha):
    x = x0
    x_values = []
    for _ in range(iters):
        x_values.append(x)
        x = x - alpha * grad_f(matrix, x)
    return x, x_values

def gradiente_descendente2(matrix, x0, iters, alpha, delta):
    x = x0
    x_values = []
    for _ in range(iters):
        x_values.append(x)
        x = x - alpha * grad_f2(matrix, x,delta)
        print(x[0])
    return x, x_values


def pseudo_inverse(A, d):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_d = np.diag(1/S[:d])
    # print(S_d.shape)
    A_d = np.dot(VT.T[:,:d], S_d).dot(U.T[:d,:])
    # print(A_d.shape)
    return A_d

def absolute_error(A,B):
    abserr = 0 
    
    for i in range(A.shape[0]):
        abserr += abs(A[i]-B[i])

    return abserr
        

def find_B(X, Y, d):
    A_d = pseudo_inverse(X, d)
    B = np.dot(A_d, Y)
    return B



def main1():
    matrix = np.random.randn(n, d)
    alpha = calculate_alpha(matrix)
    x1, x1_values = gradiente_descendente(matrix, x0, iters, alpha)
    print("After 1000 iters, F1(x*)=",F(matrix, x1))
    delta2 = 1e-2 * (np.max(np.linalg.svd(matrix, compute_uv=False)))
    # print(np.max(np.linalg.svd(matrix, compute_uv=False)))
    # print(delta2)
    x2, x2_values = gradiente_descendente2(matrix, x0, iters, alpha,delta2)
    print("After 1000 iters, F2(x*)=",F2(matrix, x2,delta2))


    b1 = find_B(matrix, b, 5)
    print("B =", b1)
    print(F(matrix, b1))
    print("Absolute error =", absolute_error(x1,b1))
    y1_values = [F(matrix, x) for x in x1_values]
    y2_values = [F2(matrix, x,delta2) for x in x2_values]
    iteraciones = range(iters)


    y1_values = [F(matrix, x) for x in x1_values]
    y2_values = [F2(matrix, x, delta2) for x in x2_values]


    plt.figure(figsize=(10, 6))
    plt.plot(y2_values, label='Descenso por gradiente con regularización L2')
    plt.plot(y1_values, label='Descenso por gradiente')
    plt.plot([F(matrix, b1)]*iters, label='F(B), least squares')

    plt.plot([F2(matrix, b1,delta2)]*iters, label='F2(B), regularized least squares')

    plt.plot([F2(x2, b1,delta2)]*iters, label='F2(x2), regularized least squares')
    plt.yscale("log")
    plt.xlabel('Iteraciones')
    plt.ylabel('Valores de x')
    plt.title('Evolución de los valores de x a lo largo de las iteraciones')
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main1()