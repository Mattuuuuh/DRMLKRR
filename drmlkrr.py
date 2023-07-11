import logging
import time
import warnings

import numpy as np
import sklearn as sk
import sklearn.model_selection
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

from scipy.linalg import lu_factor, lu_solve
import scipy.sparse.linalg
import mlkrr

#import pandas as pd
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps

class low_rank_MLKRR:
    """
    TODO
    doctest
    """
    def __init__(self, MLKRR, num_iter_fit=5, max_iter_subset_selection=5, logging=False):
        self.max_iter_subset_selection = max_iter_subset_selection
        self.num_iter_fit = num_iter_fit
        self.MLKRR = MLKRR
        self.init = MLKRR.init
        self.tol = MLKRR.tol
        self.verbose = MLKRR.verbose
        self.reg = MLKRR.krr_regularization
        self.sigma = MLKRR.sigma
        self.size_alpha = MLKRR.size_alpha
        self.size_A = MLKRR.size_A
        self.logging= logging

    def simpleloss(self,A):
        sigma = self.sigma
        reg = self.reg
        X = self.X
        y = self.y
        indices_X1 = self.indices_X1
        indices_X2 = self.indices_X2

        y1 = y[indices_X1]
        y2 = y[indices_X2]
        X1 = X[indices_X1]
        X2 = X[indices_X2]

        Xe = np.dot(X, A.T)
        X1e = Xe[indices_X1]
        X2e = Xe[indices_X2]

        kernel_constant = 1 / (1 * np.sqrt(2 * np.pi) * sigma)
        exponent_constant = 1 / (1 * sigma**2)

        dist1 = pairwise_distances(X1e, squared=True, n_jobs=-1)

        kernel1 = kernel_constant * np.exp(-dist1 * exponent_constant)

        n1 = len(X1)
        H=kernel1+reg*np.eye(n1)
        lu, pivot = lu_factor(H, check_finite=False)
        alphas=lu_solve((lu,pivot), y1, check_finite=False)

        intercept=0
        dist2 = pairwise_distances(X2e, X1e, squared=True, n_jobs=-1)

        kernel2 = kernel_constant * np.exp(-dist2 * exponent_constant)

        yhat2 = kernel2 @ alphas + intercept

        ydiff2 = yhat2 - y2
        cost = (ydiff2**2).sum()
        #return cost
        #maximize
        return -cost

    def grad(self,A):
        sigma = self.sigma
        reg = self.reg
        X = self.X
        y = self.y
        indices_X1 = self.indices_X1
        indices_X2 = self.indices_X2

        X1 = self.X1
        X2 = self.X2

        y1 = self.y1
        y2 = self.y2

        Xe = X@A.T
        X1e = Xe[indices_X1]
        X2e = Xe[indices_X2]
        n1 = len(X1)

        kernel_constant = 1 / (1 * np.sqrt(2 * np.pi) * sigma)
        exponent_constant = 1 / (1 * sigma**2)
        
        n_jobs=-1
        kernel1=pairwise_kernels(X1e, metric='rbf', gamma=exponent_constant, n_jobs=n_jobs)*kernel_constant 
        kernel2=pairwise_kernels(X2e, X1e, metric='rbf', gamma=exponent_constant, n_jobs=n_jobs)*kernel_constant 

        # LU decomposition of H used everytime H^-1 @ b or H^-T @ b is computed
        H=kernel1+reg*np.eye(n1)
        lu, pivot = lu_factor(H, check_finite=False)
        alphas=lu_solve((lu,pivot), y1, check_finite=False)

        intercept = 0
        
        yhat2 = kernel2 @ alphas + intercept
        ydiff2 = yhat2 - y2
        #cost = (ydiff2**2).sum()

        ################# GRADIENTS #################
        # matrix gradient
        u=lu_solve((lu,pivot), kernel2.T@ydiff2, trans=1, check_finite=False)
        W = ydiff2[:, np.newaxis] * kernel2 * alphas
        Q = np.diag(np.sum(W, axis=1))
        R = np.diag(np.sum(W, axis=0))
        S = kernel1 * u[:, np.newaxis] * alphas
        T = -S - S.T + np.diag(np.sum(S, axis=0) + np.sum(S, axis=1))
        s1=X2.T@(-W)@X1
        s2=X2e.T@Q@X2
        s3=X1e.T@(R-T)@X1
        gradA = -4*exponent_constant*(A@(s1+s1.T) +  s2+s3)
        
        #return gradA
        # maximize
        return -gradA

    # projects M onto tangent space at R
    def orth_proj(self,R, M):
        return 1/2 * (M - R@M.T@R)

    def riem_gradient(self,A,P):
        G=A@self.grad(P.T@A).T
        return self.orth_proj(P,G)

    def retract(self,R,G,step):
        Q=np.linalg.qr(R+step*G)[0]
        return Q

    # Line search step.
    # This is the backtracking line search by Armijo (1966), see wiki.
    # The constant 0.1 is a free parameter in this method, and I chose it randomly.
    def step(self,A,P,G,laststep):
        #if laststep>-1e-10:
        #    s=-1e-7
        #else:
        #    s=2*laststep
        s=min(2*laststep, 1e-4)
        normG=np.linalg.norm(G)
        if self.logging: print("normG", normG)
        t=(0.01)*normG**2
        k=0
        diff = self.simpleloss(P.T@A)-self.simpleloss((P+s*G).T@A)
        while diff < s*t:
            diff = self.simpleloss(P.T@A)-self.simpleloss((P+s*G).T@A)
            if self.logging: print(s, diff)
            k+=1
            s /= 2
            if s < 1e-15:
                    return s
        return s

    # Stop conditions in gradient descent.
    def check_stop(self,G,vals,k,opt_steps,tol):
        maxsteps=opt_steps
        if self.verbose: print(f"Step: {k} / {maxsteps}")
        if(np.linalg.norm(G) < tol):
            if self.verbose: print("Stop due to gradient tolerance.")
            return True
        # outcommented for now
        #if(np.abs(vals[k] - vals[k+1]) < tol):
        #    print("Stop due to value tolerance.")
        #    return True
        return k>=maxsteps 

    def orthonormal_completion(self, P):
        rank=P.shape[1]
        u, s, vh = np.linalg.svd(P)
        P=u[:,:rank]
        Porth=u[:,rank:]
        return P, Porth

    # optimizes the loss of P.T@A
    # P is n-by-k and A is n-by-n
    # we pick P to be orthogonal as well (P.T @ P = 1)
    def optimize(self,A,P):
        if self.verbose: print("Starting subspace loss maximization.")
        M=[P]
        vals=[-self.simpleloss(P.T@A)]
        
        k=0
        s=1e-4 # starting step, important choice
        stop=False
        while not stop:
            P=M[k]
           
            if self.logging: print("G")
            G=-self.riem_gradient(A,P)

            if self.logging: print("s")
            s=self.step(A,P,G,s)
            assert s>0, "step size is negative?"
            
            if self.logging: print("retract")
            newP=self.retract(P,G,s)

            M.append(newP)
            vals.append(-self.simpleloss(newP.T@A))
            if self.logging: print(-self.simpleloss(newP.T@A))

            if self.logging: print("stop")
            stop=self.check_stop(G,vals,k,opt_steps=self.max_iter_subset_selection,tol=self.tol)
            
            k=k+1
        
        return M[-1], vals

    def fit(self, X, y, rank):
        if self.verbose: print(f"Running fit... \n Rank : {rank}/{X.shape[1]}, fit iter: {self.num_iter_fit}, subspace iter: {self.max_iter_subset_selection}, MLKRR iter: {self.MLKRR.max_iter_per_shuffle}")

        self.X=X
        self.y=y

        shuffle_n_ = 1
        shuffle_index = 1

        size_alpha=self.size_alpha
        size_A=self.size_A

        self.indices_X1, self.indices_X2 = sk.model_selection.train_test_split(
            np.arange(len(X)),
            train_size=size_alpha,
            test_size=size_A,
            random_state=shuffle_index,
        )

        self.X1 = X[self.indices_X1]
        self.X2 = X[self.indices_X2]

        self.y1 = y[self.indices_X1]
        self.y2 = y[self.indices_X2]

        n, d = X.shape
        
        A=np.eye(d)
        P=np.eye(d, rank)

        losses=[-self.simpleloss(A)]
        tests=[]
        for i in range(self.num_iter_fit):
            if self.verbose: print(f"Fit iteration {i+1} / {self.num_iter_fit}")

            P, vals=self.optimize(A,P)
            if self.verbose: print("First and last local losses", vals[0], vals[-1])

            # complete P to orthonormal basis.
            P, Porth = self.orthonormal_completion(P)

            B=P.T@A
            u, s, vh = np.linalg.svd(B, full_matrices=False)
            assert len(s) == rank
            newX = X@vh.T

            MLKRR = self.MLKRR
            
            MLKRR.init=np.diag(s)

            MLKRR.fit(newX, y)
            newS=MLKRR.A

            #print(-self.simpleloss(P.T@A))

            # actually if we did optimize on positive semi-definite matrices, we would not have this problem
            # since then A = P @ diag(s) @ P.T and we would modify newX=X@P.T and optimize on diag(s)...

            # ignore u because P.T@newA = diag(newS) @ vh generates the same norm as u @ diag(newS) @ vh
            # check this in first loss evaluation optimize(P)? it makes sense though
            # removing u isn't so much faster, so we can also keep it.
            #A=P@newS@vh

            # actually P@... isn't good: A becomes rank k. We want to keep P.T info and old info
            #Porth = np.linalg.svd(P)[0][:,:rank]

            newA = P@u@newS@vh + Porth@Porth.T@A
            #
            if self.verbose: print("New local loss after MLKRR", -self.simpleloss(P.T@newA))

            # new loss
            global_loss=-self.simpleloss(newA)
            old_global_loss=losses[-1]
            if self.verbose: print("Previous global loss", old_global_loss)
            if self.verbose: print("New global loss", global_loss)
            
            if old_global_loss < global_loss:
                if self.verbose: print("New loss is not an improvement, continuing.")
                losses.append(old_global_loss)
                continue
            
            losses.append(global_loss)
            A=newA

            # test RMSE stuff
            
            # this alpha is wrong -- change
            # can't believe it compiles actually
            # also this changes global variables and messes everything up
            """
            alpha=MLKRR.alpha
            Xt_embedded = Xtest@A.T
            indices_X1=MLKRR.indices_X1
            X1e=X[indices_X1]@A.T
            kernel_constant = 1 / (1 * np.sqrt(2 * np.pi) * sigma)
            exponent_constant = 1 / (1 * sigma**2)
            kernel_test=pairwise_kernels(Xt_embedded, X1e, metric='rbf', gamma=exponent_constant, n_jobs=-1)*kernel_constant
            yhat_test = kernel_test @ alpha
            ydiff_test = np.array(yhat_test - ytest)

            test_rmse = np.sqrt(np.mean(ydiff_test**2))
            test_mae = np.mean(np.abs(ydiff_test))
            tests.append(test_rmse)
            """

        print("RMSE", tests)
        print("A loss", losses)
        plt.plot(range(len(losses)),losses)
        plt.show()

        return self
