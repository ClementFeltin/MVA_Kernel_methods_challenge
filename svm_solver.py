import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from scipy.linalg import svd

def qp(H, e, A,y, b, C=np.inf, l=1e-8, verbose=True):
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
    n = K.shape[0]
    H = y*((y*K).T)
    e = np.ones(n)
    A = y.reshape((1,-1))
    b = 0
    mu = qp(H, e, A,y, b, C, l=1e-8, verbose=False)
    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    mu_support = mu[idx_support]
    return mu_support, idx_support

def compute_b(K, y, mu_support, idx_support):
    y_support = y[idx_support]
    K_support = K[idx_support][:, idx_support]
    b = np.mean(y_support - np.sum(mu_support*y_support*K_support,axis=1))
    return b


class SVM_cvxopt(object):
    """A class for solving with cvxopt the linear SVM dual problem without intercept """

    def __init__(self,kernel,C):
        self.kernel = kernel
        self.C = C

    def fit(self,X,y):
        K = self.kernel(X,X)
        self.mu_support, self.idx_support = svm_solver(K, y,C=self.C)
        G = np.multiply(y,X.T).T
        self.y_support = y[self.idx_support]
        self.X_support = X[self.idx_support]
        self.w = np.dot(self.mu_support,G[self.idx_support])
        self.b = compute_b(K, y, self.mu_support, self.idx_support)

    def predict(self,X):
        G = self.kernel(X, self.X_support)
        decision = G.dot(self.mu_support * self.y_support) + self.b
        y_pred = np.sign(decision)
        return(y_pred)

    def predict_margin(self,X):
        G = self.kernel(X, self.X_support)
        decision = G.dot(self.mu_support * self.y_support) + self.b
        return(decision)

def coordinate_descent(mu_init, grad_i, prox_g_i, step_i, n_features, n_iter):
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

class SVM_cd(object):
    pass

class SVM_multiclass(object):

    def __init__(self,kernel,C,nb_classes,solver='cvxopt'):
        self.kernel = kernel
        self.C = C
        self.clf = []
        self.nb_classes=nb_classes

    def fit(X,y):
        self.clf=[]
        for i in range(self.nb_classes):
            y_train = (y==i)*np.ones(y.shape[0])
            if solver=='cvxopt':
                clfi = SVM_cvxopt(self.kernel,self.C)
            else:
                clfi = SVM_cd(self.kernel,self.C)
            clfi.fit(X,y_train)
            self.clf.append(clfi)

    def predict(X):
        predictions = np.asarray([self.clf[i].predict_margin(X) for i in range(self.nb_classes)])
        return np.argmax(predictions,axis=1)















    
