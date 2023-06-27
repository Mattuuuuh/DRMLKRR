# DRMLKRR

Dimension-reduced metric learning with kernel ridge regression.

Given a data set $X$, labels $y$, and a rank $k$, DRMLKRR iteratively computes the worst rank-$k$ subset of the feature space on which to apply MLKRR.

From a starting matrix $A \in \mathbb{R}^{n \times n}$, each step of `fit` computes an orthogonal matrix $P \in \mathbb{R}^{n \times k}$.
The matrix $P^T A$ defines a map from $\mathbb{R}^n$ to $\mathbb{R}^k$ which induces a low-rank metric.
$P$ is search for such that the loss $\ell (P^T A)$ is maximized.

SVD gives $P^T A = U \Sigma V$ such that $U, \Sigma$ are $k \times k$, and $V$ is $k \times n$. Setting $\tilde{x} = Vx$ for each data point $x \in X$, we project the data-points onto the $k$-dimensional space, on which we apply MLKRR.

We apply MLKRR on data $\tilde{X}$ and labels $y$ with starting point $\Sigma$ to get a new matrix $\Theta$.
We also complete the basis of $P$ into an orthonormal basis of $\mathbb{R}$, producing $P^\perp \in \mathbb{R}^{n \times n-k}$.
This is done in order to keep old information of $A$ (almost) intact.

To this end, we define $\tilde{A} = P \cdot (U \Theta V) + P^\perp (P^\perp)^T A$: the new metric projected onto the space spanned by $P$, plus the projection of $A$ onto the orthogonal space.
This new matrix verifies
$$P^T \tilde{A} = U \Theta V$$
which gives $P^T \tilde{A} x = U \Theta \tilde{x}$.
$U$ being an orthonormal matrix, it can be safely ignored when considering the induced norm of $P^T \tilde{A}$. In fact we might be able to remove it entirely?

This shows that the new metric $\tilde{A}$ projects optimally onto the subspace defined by $P$, since $\Theta$ gives the optimal metric with respect to data-set $\tilde{X}$ (by MLKRR).

# Dependencies 

Same as MLKRR (see https://github.com/lcmd-epfl/MLKRR), and MLKRR itself.

# Examples

File `dim_red_example.py` gives an example of DRMLKRR.
File `mlkrr_example.py` gives an example of MLKRR.
