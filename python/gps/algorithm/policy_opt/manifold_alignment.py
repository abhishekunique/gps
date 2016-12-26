import numpy as np
import scipy
import math

from sklearn.neighbors import NearestNeighbors
import itertools as it

class UMA:
    def __init__(self, n_components, k=2):
        self.k = k
        self.n_components = n_components
        

    def align(self, X,Y):
        
        Xnb = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(X)
        Ynb = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(Y)
        print "Getting Rx"
        Rx = self.getR(X, Xnb)
        print "Getting Ry"
        Ry = self.getR(Y, Ynb)

        # Need to make function return Dx
        print "Getting Wx"
        Wx, Dx = self.get_distances(Rx,Rx)
        print "Getting Wy"
        Wy, Dy = self.get_distances(Ry,Ry)
        print "Getting W"
        W, Gamma1 = self.get_distances(Rx,Ry)
        Gamma2 = W
        Gamma3 = W.T
        Gamma4 = np.zeros((Ry.shape[0], Ry.shape[0]))
        for i in range(Ry.shape[0]):
            Gamma4[i,i] = np.sum(Gamma3[i,:])

        print "Ls"
        Lx = Dx-Wx
        Ly = Dy-Wy
        print "Zs and Ds"
        Z = np.zeros((X.shape[0]+Y.shape[0], X.shape[1]+Y.shape[1]))
        Z[:X.shape[0], :X.shape[1]] = X
        Z[X.shape[0]:, X.shape[1]:] = Y
        D = np.zeros((Dx.shape[0]+Dy.shape[0], Dx.shape[1]+Dy.shape[1]))
        D[:Dx.shape[0], :Dx.shape[1]] = Dx
        D[Dx.shape[0]:, Dx.shape[1]:] = Dy

        # What is mu? Who knows!
        mu= 1
        print "Big L"
        L = np.zeros((Lx.shape[0]+Ly.shape[0], Lx.shape[1]+Ly.shape[1]))
        L[:Lx.shape[0],:Lx.shape[1]] = Lx+mu*Gamma1
        L[:Lx.shape[0], Lx.shape[1]:] = -1*mu*Gamma2
        L[Lx.shape[0]:,:Lx.shape[1]] =  -1*mu*Gamma3
        L[Lx.shape[0]:, Lx.shape[1]:] =Ly+mu*Gamma4


        # Z is transpose form in the paper
        A=np.dot(np.dot(Z.T,D), Z)
        M=np.dot(np.dot(Z.T,L), Z)
        # General eigenvalue decomposition
        print "Eigs"
        eigvals, eigvecs = scipy.sparse.linalg.eigs(A=A, M=M, k=self.n_components, which='SM')
        self.alpha = eigvecs
        self.alpha_s = eigvecs[:X.shape[1]]
        self.alpha_t = eigvecs[X.shape[1]:]
        import IPython
        IPython.embed()
        self.t_to_s = np.dot(np.linalg.pinv(self.alpha_t.T), self.alpha_s.T)

    def getR(self,X, Xnb):
        R = np.zeros((X.shape[0], self.k, self.k))
        for xi in range(X.shape[0]):
            distances, indices = Xnb.kneighbors(X[xi,:])
            for i in range(indices):
                for j in range(indicies):
                    R[xi,i,j] = np.linalg.norm(X[i]-X[j])
        return R

    def get_distances(self, Rx,Ry):
        D = np.zeros((Rx.shape[0], Rx.shape[0]))

        W = np.zeros((Rx.shape[0], Ry.shape[0]))
        for i in range(Rx.shape[0]):
            for j in range(Ry.shape[0]):
                W[i,j] = math.exp(-self._d(Rx[i], Ry[j]))
        for i in range(Rx.shape[0]):
            D[i,i] = np.sum(W[i,:])
        return W, D

    def _d(self, Rxi,Rxj):
        bestd = float('inf')
        perms = it.permutations(range(self.k))
        for p in perms:
            Rxjp = self. get_perm(Rxj, p)
            gamma1 = np.trace(np.dot(Rxi.T, Rxjp))/np.trace(np.dot(Rxi.T, Rxi))
            gamma2 = np.trace(np.dot(Rxjp.T,Rxi))/np.trace(np.dot(Rxjp.T, Rxjp))
            d = min(np.linalg.norm(Rxjp-gamma1*Rxi, ord='fro'),
                    np.linalg.norm(Rxi-gamma2*Rxjp, ord='fro'))
            if d < bestd:
                bestd = d
        return bestd

    def get_perm(self,R, p):
        p = np.asarray(p)
        Rp = R[p]
        Rpp = Rp[:,p]
        return Rpp

