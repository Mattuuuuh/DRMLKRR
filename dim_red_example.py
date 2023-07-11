# Applies MLKRR on a random subset of size 10 000 of fragments of qm9, to learn their energies. 
# The initial set is split into 10 000 fragments for the learning, and 2 000 for test.

# The data points are FCHL representations (vectors of dimension 720), and the labels are the associated energies (u0)
# The FCHL data fchls_glob_qm9.npy is available at [??]

# At each iteration of the minimization algorithm (possibly multiple steps before making progess),
# the predictions are compared with the labels, appending test_maes, test_rmses, and train_maes, train_rmses.

# Before that, the variance sigma is optimized if learn_sigma is set to True.
# The total number of iterations is equal to the product of shuffles with max_iter_per_shuffle.

# For a data set of 20 000, each iteration takes up to 20 seconds. Optimizing sigma takes an additional 200 seconds.
# It should take around 5 hours for 900 iterations (eg. 30 shuffles, 30 iterations per shuffle).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlkrr
import drmlkrr as drm

X=np.load("fchls_glob_qm9.npy", allow_pickle=True)
m, n=X.shape
y=np.load("u0.npy", allow_pickle=True)

# takes 4000 random indices, and 2000 random indices among them for data and test
S=8000
indices=np.random.choice(range(m), size=S, replace=False)
indices_test=np.random.choice(range(S), size=2000, replace=False)
mask=np.ones(S, dtype=bool)
mask[indices_test]=0
ind_test=indices[np.logical_not(mask)]
ind_data=indices[mask]
print(f"Size of learning data: {len(ind_data)}. \n Size of testing data: {len(ind_test)}")

# initialize parameters
M = mlkrr.MLKRR(
        size_A=0.5, 
        size_alpha=0.5,
        verbose=True,
        shuffle_iterations=1,
        max_iter_per_shuffle=20, 
        #test_data=[X[ind_test],y[ind_test]], #TODO: implement tests
        sigma=55.0,
        krr_regularization=1e-9
        )

LRM = drm.low_rank_MLKRR(M, num_iter_fit=20, max_iter_subset_selection=10, logging=True)

# run optimization and save object
LRM.fit(X[ind_data], y[ind_data], rank=100)
np.save("DRMLKRR.npy",LRM)

