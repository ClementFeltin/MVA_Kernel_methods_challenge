import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvx
from scipy.linalg import svd

def qp(H, e, A, b, C=np.inf, l=1e-8, verbose=True):
    # Gram matrix
    n = H.shape[0]
    H = cvxopt.matrix(H)
    A = cvxopt.matrix(y, (1, n))
    e = cvxopt.matrix(-e)
    b = cvxopt.matrix(0.0)
    if C == np.inf:
        G = cvxopt.matrix(np.diag(np.ones(n) * -1))
        h = cvxopt.matrix(np.zeros(n))
    else:
        G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * -1),
                                         np.diag(np.ones(n))], axis=0))
        h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    solution = cvxopt.solvers.qp(H, e, G, h, A, b)

    # Lagrange multipliers
    mu = np.ravel(solution['x'])
    return mu

def svm_solver(K, y, C=np.inf):
    n = X.shape[0]
    H = y*((y*K).T)
    e = np.ones(n)
    A = y.reshape((1,-1))
    b = 0
    mu = qp(H, e, A, b, C, l=1e-8, verbose=False)
    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    mu_support = mu[idx_support]
    return mu_support, idx_support

def compute_b(K, y, mu_support, idx_support):
    y_support = y[idx_support]
    K_support = K[idx_support][:, idx_support]
    b = np.mean(y_support - np.sum(mu_support*y_support*K_support,axis=1))
    return b

def coordinate_descent(mu_init, grad_i, prox_g_i, step_i, n_features, n_iter, callback =None):
    mu= mu_init.copy()
    mu_new= mu_init.copy()

    for k in range(n_iter):
        i = np.random.randint(n_features)
        grad = grad_i(mu,i)
        mu_new[i] -= grad/model.H[i,i]
        mu[i] = prox_g_i(mu_new,i)

    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    mu_support = mu[idx_support]
    return mu_support, idx_support

class SVM_cvxopt(object):
    """A class for solving with cvxopt the linear SVM dual problem without intercept """

    def __init__(self,kernel,C):
        self.kernel = kernel
        self.C = C

    def fit(X,y):
        self.K = self.kernel(X,X)
        self.mu_support, self.idx_support = svm_solver(self.K, y)
        self.G = np.multiply(y,X.T).T
        self.y_support = y[self.idx_support]
        self.X_support = X[self.idx_support]
        self.w = np.dot(self.mu_support,self.G[self.idx_support])
        self.b = compute_b(K, y, self.mu_support, self.idx_support)

    def predict(X):
        G = kernel(X, self.X_support)
        decision = G.dot(self.mu_support * self.y_support) + self.b
        y_pred = np.sign(decision)
        return(y_pred)
    
