import numpy as np

if __name__ == '__main__':
    for k in range(50):
        l = k * 2 + 1
        A = np.ones((l,l))
        A[:k,:k] = np.eye(k)
        A[:k,k:2*k] = -np.eye(k)
        A[k:2*k,:k] = -np.eye(k)
        A[k:2*k,k:2*k] = np.eye(k)
        A[-1,-1] = k
        # print(A)

        eigenvalues, eigenvectors = np.linalg.eig(A)
        print('k: {}, max eigenvalue: {}'.format(k, eigenvalues))

