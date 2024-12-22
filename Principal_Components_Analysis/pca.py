import numpy as np

def pca(x, threshold, method = "SVD"):
# x: original data, matrix
# threshold: threshold for contribution rate, scalar
# method: method taken, ED(eigenvalue decomposition) or SVD(singular value
#         decomposition), optional

    x = np.array(x)

    # prevent error on non-square
    rx, cx = x.shape
    if rx != cx and method == "ED":
        method = "SVD"
        print("Warning: ED can't work for non-square matrix. Method switched to SVD.")
    
    # standardization
    x_mean = np.mean(x, axis = 0)
    x_std = np.std(x, axis = 0, ddof = 0)
    x_centered = (x - x_mean) / x_std

    if method == "ED":
        # covariance matrix
        R = np.cov(x_centered, rowvar = False)
        
        # eigenvalue and eigenvector
        eig_value, V = np.linalg.eigh(R)

        idx = np.argsort(eig_value)[::-1]
        eig_value = eig_value[idx]
        V = V[:, idx]

    elif method == "SVD":
        # singular value decomposition
        U, S, Vt = np.linalg.svd(x_centered, full_matrices = False)

        # sort eigenvalues
        eig_value = np.diag(S)**2
        idx = np.argsort(eig_value)[::-1]
        eig_value = eig_value[idx]
        V = Vt.T[:, idx]

    else:
        raise ValueError("Invalid method. Choose either 'ED' or 'SVD'.")

    # contribution rate
    sum_eig_value = np.sum(eig_value)
    ctd_rate = np.cumsum(eig_value / sum_eig_value)
    taken_num = np.searchsorted(ctd_rate, threshold) + 1

    # calculate principal components
    Z = x_centered @ V
    Z = Z[:, :taken_num]
    return Z



