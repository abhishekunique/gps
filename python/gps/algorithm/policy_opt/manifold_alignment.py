import numpy as np
import scipy
import scipy.io
import math
import time
from sklearn.neighbors import NearestNeighbors
import itertools as it

class UMA:
    def __init__(self, n_components, k=2):
        self.k = k
        self.n_components = n_components
        

    def align(self, X,Y):

        self.meanX = np.mean(X, axis=0)
        self.meanY = np.mean(Y, axis=0)
        X = X-self.meanX
        Y = Y-self.meanY

        self.varX = np.std(X, axis=0)
        self.varY = np.std(Y, axis=0)
        X = X/self.varX
        Y = Y/self.varY
        import scipy.io
        scipy.io.savemat("multiproxy.mat",{'X': X, 'Y':Y})

        # Xnb = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(X)
        # Ynb = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(Y)
        # print "Getting Rx"
        # Rx = self.getR(X, Xnb)
        # np.save('Rx.npy', Rx)
        # print "Getting Ry"
        # Ry = self.getR(Y, Ynb)
        # np.save('Ry.npy', Ry)
        # # Need to make function return Dx
        # print "Getting Wx"
        # t0 = time.clock()
        # Wx, Dx = self.get_distances(Rx,Rx)
        # t1 = time.clock()
        # np.save('Wx.npy', Wx)
        # print "time for Wx", t1-t0
        # print "Getting Wy"
        # Wy, Dy = self.get_distances(Ry,Ry)
        # t2 = time.clock()
        # np.save('Wy.npy', Wy)
        # print "time for Wy", t2-t1
        # print "Getting W"
        # W, Gamma1 = self.get_distances(Rx,Ry)
        # t3 = time.clock()
        # print "Time for W", t3-t2
        # np.save('W.npy', W)

        # Rx = np.load('Rx.npy')
        # Ry = np.load('Ry.npy')
        # W = np.load('W.npy')
        # Wy = np.load('Wy.npy')
        # Wx = np.load('Wx.npy')
        # Dx = self.get_D(Wx, Rx, Rx)
        # Dy = self.get_D(Wx, Ry, Ry)
        # Gamma1 = self.get_D(W, Rx, Ry)
        # Rx = scipy.io.loadmat('Rx.mat')['Dist1']
        # Ry = scipy.io.loadmat('Ry.mat')['Dist2']
        # Wx = scipy.io.loadmat('W_X.mat')['W11'].toarray()
        # Wy = scipy.io.loadmat('W_Y.mat')['W22'].toarray()
        # W = scipy.io.loadmat('W_eeparticle.mat')['W']
        # Dx = self.get_D(Wx, Rx, Rx)
        # Dy = self.get_D(Wx, Ry, Ry)
        # Gamma1 = self.get_D(W, Rx, Ry)

        # Gamma2 = W
        # Gamma3 = W.T
        # Gamma4 = np.zeros((Ry.shape[0], Ry.shape[0]))
        # for i in range(Ry.shape[0]):
        #     Gamma4[i,i] = np.sum(Gamma3[i,:])

        # print "Ls"
        # Lx = Dx-Wx
        # Ly = Dy-Wy
        # print "Zs and Ds"
        # Z = np.zeros((X.shape[0]+Y.shape[0], X.shape[1]+Y.shape[1]))
        # Z[:X.shape[0], :X.shape[1]] = X
        # Z[X.shape[0]:, X.shape[1]:] = Y
        # D = np.zeros((Dx.shape[0]+Dy.shape[0], Dx.shape[1]+Dy.shape[1]))
        # D[:Dx.shape[0], :Dx.shape[1]] = Dx
        # D[Dx.shape[0]:, Dx.shape[1]:] = Dy

        # # What is mu? Who knows!

        # mu= 1
        # print "Big L"
        # L = np.zeros((Lx.shape[0]+Ly.shape[0], Lx.shape[1]+Ly.shape[1]))
        # L[:Lx.shape[0],:Lx.shape[1]] = Lx+mu*Gamma1
        # L[:Lx.shape[0], Lx.shape[1]:] = -1*mu*Gamma2
        # L[Lx.shape[0]:,:Lx.shape[1]] =  -1*mu*Gamma3
        # L[Lx.shape[0]:, Lx.shape[1]:] =Ly+mu*Gamma4


        # # Z is transpose form in the paper
        # A=np.dot(np.dot(Z.T,D), Z)
        # M=np.dot(np.dot(Z.T,L), Z)
        # # General eigenvalue decomposition
        # print "Eigs"
        # # import IPython
        # # IPython.embed()

        # eigvals, eigvecs = scipy.sparse.linalg.eigs(A=A, M=M, k=self.n_components*2, which='SM')
        # # norm_eigvecs = np.linalg.norm(eigvecs, axis=0)
        # # for v in range(eigvecs.shape[0]):
        # #     eigvecs[v] = eigvecs[v]/norm_eigvecs[v]
        # self.alpha = eigvecs
        # self.alpha_s = eigvecs[:X.shape[1]]
        # self.alpha_t = eigvecs[X.shape[1]:]

        self.alpha_s = scipy.io.loadmat('map1_mult_knn.mat')['map1']
        self.alpha_t = scipy.io.loadmat('map2_mult_knn.mat')['map2']
        # import IPython
        # IPython.embed()
        self.t_to_s = np.dot(np.linalg.pinv(self.alpha_t.T), self.alpha_s.T)
        # self.t_to_s needs to be normalized
        norm = np.linalg.norm(self.t_to_s, axis=0)
        norm_eigvecs = np.linalg.norm(self.t_to_s, axis=0)
        for v in range(self.t_to_s.shape[1]):
            self.t_to_s[:,v] = self.t_to_s[:,v]/norm_eigvecs[v]


    def apply(Y):
        return np.dot(Y, self.t_to_s)

    def getR(self,X, Xnb):
        R = np.zeros((X.shape[0], self.k, self.k))
        for xi in range(X.shape[0]):
            distances, indices = Xnb.kneighbors(X[xi,:], self.k)
            indices = indices[0]
            # print "X", X.shape, "xi", xi, "k", self.k, "ind", len(indices), indices
            for i in range(self.k):
                for j in range(self.k):
                    R[xi,i,j] = np.linalg.norm(X[indices[i]]-X[indices[j]])
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

    def get_D(self, W, Rx, Ry):
        D = np.zeros((Rx.shape[0], Rx.shape[0]))
        for i in range(Rx.shape[0]):
            D[i,i] = np.sum(W[i,:])
        return D

    def _d(self, Rxi,Rxj):
        bestd = float('inf')
        perms = it.permutations(range(self.k))
        for p in perms:
            Rxjp = self.get_perm(Rxj, p)
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

    def plot(self, X, Y, eeX, eeY):
        import matplotlib.pyplot as plt
        newY = (Y-self.meanY)/self.varY
        Xhat = np.dot(newY, self.t_to_s)*self.varX+self.meanX
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(Xhat)
        distances_2, indices_2 = nbrs.kneighbors(Xhat)

        # indices = np.reshape(indices, (num_conds, num_samples, T_extended))
        # dO_robot0 = obs_extended_full[0].shape[-1]
        # obs_full_reshaped = np.reshape(obs_extended_full[0], (num_conds*num_samples*T_extended,dO_robot0))
        # for i in range(X.shape[0]):
        #     x = eeX[i][0]
        #     y = eeX[i][2]
        #     nnbr_currpoint = indices[i][0]
        #     x_nbr = eeY[nnbr_currpoint][0]
        #     y_nbr = eeY[nnbr_currpoint][2]
        #     print("X: " + str([x,x_nbr]))
        #     print("Y: " + str([y,y_nbr]))
        # #     lines = plt.plot([x,x_nbr], [y,y_nbr])
        # # import IPython
        # # IPython.embed()
        # # plt.show()
        # for i in range(X.shape[0]):
        #     x = eeX[i][0]
        #     y = eeX[i][2]
        #     nnbr_currpoint = i#indices[i][0]
        #     x_nbr = eeY[nnbr_currpoint][0]
        #     y_nbr = eeY[nnbr_currpoint][2]
        #     print("X: " + str([x,x_nbr]))
        #     print("Y: " + str([y,y_nbr]))
        #     #lines = plt.plot([x,x_nbr], [y,y_nbr])
        plt.scatter(Xhat[:,0], Xhat[:,1], c='r')
        #plt.scatter(Y[:,0], Y[:,1], c='g')

        plt.scatter(X[:,0], X[:,1], c='b')
        for i in range(X.shape[0]):
            # lines = plt.plot([Y[i,0],Xhat[i,0]], [Y[i,1],Xhat[i,1]])
            y1 = newY[i]
            y = Y[i]
            x = X[i]
            xhat = np.dot(y1, self.t_to_s)*self.varX+self.meanX
            plt.scatter(xhat[0], xhat[1], c='r')
            lines = plt.plot([x[0],xhat[0]], [x[1],xhat[1]])
        import IPython
        IPython.embed()
