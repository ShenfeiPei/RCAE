cimport numpy as np
import numpy as np
np.import_array()

import sys
from cython.parallel import prange
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances as EuDist2

from fun cimport EProjSimplex_new_M

def norm_W(A):
    d = np.sum(A, 1)
    d[d == 0] = 1e-6
    d_inv = 1 / np.sqrt(d)
    tmp = A * np.outer(d_inv, d_inv)
    A2 = np.maximum(tmp, tmp.T)
    return A2

class RCAE(object):
    def __init__(self, W0, c):
        self.W0 = W0
        self.N = self.W0.shape[0]
        self.c_true = c
        self.zr = 1e-10

    def clu(self, lam=0.1, lam_s = 0.1, NITER=30):

        W = norm_W(self.W0)
        S = W.copy()


        F, _ = _WHH((W+lam_s*S)/(1+lam_s), self.c_true, 0.5, ITER=200)
        FF = F@F.T

        # early stop ?
        Ls = self.get_lap(S)
        val, vec = np.linalg.eigh(Ls)
        P = vec[:, :self.c_true]
        for i in range(self.c_true):
            if P[0, i] < 0:
                P[:, i] = -1 * P[:, i]

        if np.sum(val[:(self.c_true+1)]) < self.zr:
            raise BaseException("The original graph has more than {} connected component".format(self.c_true))
        elif np.sum(val[:self.c_true]) < self.zr:
            print("early stop")
            self._final_graph = S
        else:
            ###################### OPT #######################
            S = self.opt(W, S, F, FF, P, lam, lam_s, NITER)
            self._final_graph = S

        _, y = sp.csgraph.connected_components(csgraph=self._final_graph, directed=False, return_labels=True)
        self.Y = y


    def get_lap(self, A):
        A = np.maximum(A, A.T)
        D = np.diag(np.sum(A, axis=0))
        L = D - A
        return L

    def opt(self, W, S, F, FF, P, lam, lam_s, NITER):

        cdef int Iter = 0
        cdef int i = 0

        cdef np.ndarray[double, ndim=2] dist = np.zeros((self.N, self.N), dtype=np.float64)
        cdef np.ndarray[double, ndim=2] AD = np.zeros((self.N, self.N), dtype=np.float64)
        cdef np.ndarray[double, ndim=2] D = np.zeros((self.N, self.N), dtype=np.float64)
        cdef np.ndarray[double, ndim=2] L = np.zeros((self.N, self.N), dtype=np.float64)

        for Iter in range(NITER):
            
            # update S
            dist = EuDist2(P, P, squared=True)
            AD = FF - 0.5*lam*dist/lam_s
            tmp = EProjSimplex_new_M(AD, 1)
            S = np.array(tmp)
            #  S = np.maximum(S, S.T)

            # update P
            P_old = P.copy()

            Ls = self.get_lap(S)

            #  val, vec = self.eigsh(Ls, self.c_true + 1)
            #  Ls_sp = sp.csr_matrix(Ls)
            #  val, vec = sp.linalg.eigsh(Ls_sp, k=self.c_true+1, which="SA")
            val, vec = np.linalg.eigh(Ls)

            P = vec[:, :self.c_true]

            for i in range(self.c_true):
                if P[0, i] < 0:
                    P[:, i] = -1 * P[:, i]

            # stop ?
            fn1 = np.sum(val[:self.c_true])
            fn2 = np.sum(val[:(self.c_true + 1)])
            if fn1 > self.zr:
                lam = 2*lam
            elif fn2 < self.zr:
                lam = lam/2
                P = P_old.copy()
            else:
                break

            # update F
            F, _ = _WHH((W+ lam_s * S)/(lam_s + 1), self.c_true, 0.5, ITER=200)
            FF = F@F.T

        return S

    @property
    def y_pre(self):
        return self.Y

    @property
    def init_graph(self):
        return self._init_graph

    @property
    def final_graph(self):
        return self._final_graph

    @property
    def ref(self):
        title = "RCAE, ICASSP, 2021"
        return title

cdef _WHH(W, int c, double beta=0.5, int ITER=100):
    cdef int N = W.shape[0]
    cdef np.ndarray[double, ndim=2] H = np.zeros((N, c), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] H_old = np.zeros((N, c), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] WH = np.zeros((N, c), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] HHH = np.zeros((N, c), dtype=np.float64)

    val, vec = np.linalg.eigh(W)
    W = sp.csr_matrix(W)

    H = vec[:, -c:]
    #  H = sparse.linalg.eigsh(W, which='LA', k=c)[1]
    #  print(np.mean(H))
    H = np.maximum(W @ H, 0.00001)

    obj = np.zeros(ITER)
    obj[0] = np.linalg.norm(W - H@H.T, ord="fro")

    cdef int i = 0
    for i in range(1, ITER):
        H_old = H.copy()

        WH = W@H
        HHH = H@(H.T@H)
        H = H*(1 - beta + beta*WH/HHH)

        obj[i] = np.linalg.norm(W - H@H.T, ord="fro")

        if np.abs(obj[i] - obj[i-1])/obj[i] < 1e-6:
            break

    return H, obj
