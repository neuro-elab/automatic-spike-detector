from nimfa.utils.linalg import *


class Nmfsc:
    """
    This class provides nonegative matrix factorization with sparseness constraints. The implementation is based on
    the matlab library [1]_ developed by Hoyer, which defined sparseness in terms of a relation between the
    :math:`L_1` norm and the :math:`L_2` norm [2]_.

    Parameters
    ----------

    V: numpy.ndarray[numpy.dtype[float]]
        A matrix representing the input data.

    W: numpy.ndarray[numpy.dtype[float]], optional
        If given, these values will be used to initialize the :math:`W` matrix.

    H: numpy.ndarray[numpy.dtype[float]], optional
        If given, these values will be used to initialize the :math:`H` matrix.

    rank: int, optional, default = 5
        The rank of the decomposition.

    max_iter: int, optional, default = 10
        The number of maximally performed matrix factorization iterations if convergence is not achieved in one
        of the preceding iterations.

    min_residuals: float, optional default = 1e-4
        The maximum reconstruction error to reach convergence.

    n_runs: int, optional, default = 1
        The number of complete nmf runs performed, each run includes a new initialization of the :math:`W`
        and :math:`H` matrices.

    version: str, optional, default = 'l'
        If version = 'l', sparseness will be imposed on the columns of :math:`W`, if version = 'r',
        sparseness will be imposed on the rows of :math:`H`.

    sH: float, optional, default = 0.25
        The sparseness imposed on each row of :math:`H`, in case version = 'r'.

    sW: float, optional, default = 0.25
        The sparseness imposed on each column of :math:`W`, in case version = 'l'.

    References
    ----------
    [1] Patrick Hoyer. Nmfpack. https://github.com/aludnam/MATLAB/tree/master/nmfpack.
    [2] Patrik O Hoyer. 'Non-negative matrix factorization with sparseness constraints'
        Journal of Machine Learning Research  5:1457-1469, 2004.
    """

    def __init__(
        self,
        V,
        W=None,
        H=None,
        rank=5,
        max_iter=10,
        min_residuals=1e-4,
        n_runs=1,
        version="l",
        sH=0.25,
        sW=0.25,
    ):
        self.V = np.asmatrix(V)
        self.W = W
        self.H = H
        self.rank = rank
        self.max_iter = max_iter
        self.n_runs = n_runs
        self.version = version
        self.sH = sH
        self.sW = sW
        self.min_residuals = 1e-4 if not min_residuals else min_residuals
        self.p_c = int(ceil(1.0 / 5 * self.V.shape[1]))
        self.p_r = int(ceil(1.0 / 5 * self.V.shape[0]))

    def __call__(self):
        """Run the specified MF algorithm."""
        return self.factorize()

    def __project(
        self, s: np.ndarray[np.dtype[float]], k1: float, k2: float
    ) -> np.ndarray[np.dtype[float]]:
        # Save original shape
        shape = s.shape

        # Transform to ndarray
        s = np.asarray(s).squeeze()

        # Problem dimension
        N = len(s)

        # Start by projecting the point to the sum constraint hyperplane
        v = s + (k1 - sum(s)) / N

        # Initialize zerocoeff (initially, no elements are assumed zero)
        zero_coeff = []

        j = 0
        while True:
            # Perform projection operation
            mid_point = np.ones(N) * k1 / (N - len(zero_coeff))
            mid_point[zero_coeff] = 0
            w = v - mid_point

            # Set v = v + alpha * w, solve quadratic equation to choose alpha
            # s.t resulting v satisfies L2 norm constraint
            a = sum(np.power(w, 2))
            b = 2 * np.dot(w, v)
            c = sum(np.power(v, 2)) - k2

            alpha = (-b + np.real(np.sqrt(b**2 - 4 * a * c))) / (2 * a)
            v = v + alpha * w

            if np.all(v >= 0):
                # Found solution
                # print(f"found solution after j={j} iterations")
                break

            j = j + 1

            # Set negs to zero, subtract appropriate amount from rest
            zero_coeff = np.nonzero(v <= 0)[0]
            v[zero_coeff] = 0
            temp_sum = sum(v)
            v = v + (k1 - temp_sum) / (N - len(zero_coeff))
            v[zero_coeff] = 0

        return np.reshape(v, shape)

    def factorize(self):
        """
        Compute matrix factorization.

        Return fitted factorization model.
        """
        for run in range(self.n_runs):
            # Create initial matrices
            prng = np.random.RandomState()
            self.W = np.mat(np.zeros((self.V.shape[0], self.rank)))
            self.H = np.mat(np.zeros((self.rank, self.V.shape[1])))
            for i in range(self.rank):
                self.W[:, i] = self.V[
                    :, prng.randint(low=0, high=self.V.shape[1], size=self.p_c)
                ].mean(axis=1)
                self.H[i, :] = self.V[
                    prng.randint(low=0, high=self.V.shape[0], size=self.p_r), :
                ].mean(axis=0)
            self.H = elop(
                self.H,
                repmat(
                    np.sqrt(np.sum(multiply(self.H, self.H), axis=1)).T,
                    self.V.shape[1],
                    1,
                ).T,
                div,
            )

            # Make initial matrices have correct sparseness
            if self.version == "l":
                self.L1a = (
                    np.sqrt(self.V.shape[0]) - (np.sqrt(self.V.shape[0]) - 1) * self.sW
                )
                for idx in range(self.rank):
                    self.W[:, idx] = self.__project(
                        self.W[
                            :,
                            idx,
                        ],
                        self.L1a,
                        1,
                    )
            if self.version == "r":
                self.L1s = (
                    np.sqrt(self.V.shape[1]) - (np.sqrt(self.V.shape[1]) - 1) * self.sH
                )
                for idx in range(self.rank):
                    self.H[idx, :] = self.__project(
                        self.H[
                            idx,
                            :,
                        ],
                        self.L1s,
                        1,
                    )
            # Initial step sizes
            self.step_size_W = 1
            self.step_size_H = 1

            # Calculate initial objective
            self.p_obj = c_obj = self.objective(self.W, self.H)
            self.best_obj = c_obj if run == 0 else self.best_obj

            # Perform optimization
            iter = 0
            while self.is_satisfied(self.p_obj, c_obj, iter):
                iter += 1
                self.update()
                # self._adjustment()
                self.p_obj = c_obj
                c_obj = self.objective(self.W, self.H)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= self.best_obj or run == 0:
                self.best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                # mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        return self

    def is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping
        parameters and objective function value.

        Return logical value denoting factorization continuation.

        Parameters
        ----------
        p_obj: float
            Objective function value from previous iteration.

        c_obj: float
            Current objective function value.

        iter: int
            Current iteration number.

        Returns
        -------
        bool
            Boolean indication whether stopping criteria is met.
        """
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.min_residuals and iter > 0 and p_obj - c_obj < self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        return True

    def objective(
        self, W: np.ndarray[np.dtype[float]], H: np.ndarray[np.dtype[float]]
    ) -> float:
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.V - dot(W, H)
        return 0.5 * multiply(R, R).sum()

    def update(self):
        """
        Performs the update steps on :math:`W` and :math:`H`.
        """
        # Update H
        if self.version == "r":
            # Gradient for H
            dH = dot(self.W.T, dot(self.W, self.H) - self.V)

            begin_obj = self.p_obj

            # Make sure we decrease the objective!
            while True:
                # Take step in direction of negative gradient, and project
                H_new = self.H - self.step_size_H * dH
                for idx in range(self.rank):
                    H_new[idx, :] = self.__project(
                        H_new[
                            idx,
                            :,
                        ],
                        self.L1s,
                        1,
                    )

                # Calculate new objective
                new_obj = self.objective(self.W, H_new)

                # If the objective decreased, we can continue...
                if new_obj <= begin_obj:
                    break

                # ...else decrease step size and try again
                self.step_size_H = self.step_size_H / 2

                if self.step_size_H < 1e-200:
                    # Algorithm converged
                    return

            # Slightly increase the step size
            self.step_size_H = self.step_size_H * 1.2
            self.H = H_new

        else:
            # Update using standard NMF multiplicative update rule
            self.H = multiply(
                self.H,
                elop(dot(self.W.T, self.V), dot(self.W.T, dot(self.W, self.H)), div),
            )

            # Renormalize so rows of H have constant energy
            norms = np.sqrt(np.sum(np.power(self.H, 2), axis=1))
            self.H = elop(self.H, repmat(norms, 1, self.V.shape[1]), div)
            self.W = multiply(self.W, repmat(norms.T, self.V.shape[0], 1))

        # Update W
        if self.version == "l":
            # Gradient for W
            dW = dot(dot(self.W, self.H) - self.V, self.H.T)

            begin_obj = self.objective(self.W, self.H)

            # Make sure we decrease the objective!
            while True:
                # Take step in direction of negative gradient, and project
                W_new = self.W - self.step_size_W * dW
                norms = np.sqrt(np.sum(np.power(W_new, 2), axis=0))
                for idx in range(self.rank):
                    W_new[:, idx] = self.__project(
                        W_new[
                            :,
                            idx,
                        ],
                        self.L1a * norms[0, idx],
                        norms[0, idx] ** 2,
                    )

                # Calculate new objective
                new_obj = self.objective(W_new, self.H)

                # If the objective decreased, we can continue...
                if new_obj <= begin_obj:
                    break

                # ...else decrease step size and try again
                self.step_size_W = self.step_size_W / 2

                if self.step_size_W < 1e-200:
                    # Algorithm converged
                    return

            # Slightly increase the step size
            self.step_size_W = self.step_size_W * 1.2
            self.W = W_new

        else:
            # Update using standard NMF multiplicative update rule
            self.W = multiply(
                self.W,
                elop(
                    dot(self.V, self.H.T),
                    dot(self.W, dot(self.H, self.H.T)),
                    div,
                ),
            )

    def basis(self):
        """Return the matrix of basis vectors."""
        return self.W

    def target(self):
        """
        Return the target matrix to estimate.
        """
        return self.V

    def coef(self):
        """
        Return the matrix of mixture coefficients.
        """
        return self.H

    def connectivity(self, W=None):
        """
        Compute the connectivity matrix for the basis functions based on their coefficients.

        The connectivity matrix C is a symmetric matrix which shows the shared membership of the basis vectors:
        entry C_ij is 1 iff basis component i and basis component j belong to the same cluster, 0 otherwise.
        Basis component assignment is determined by its largest metagene expression value.

        Return connectivity matrix.
        """
        V = self.target()
        W = self.basis() if W is None else W
        _, idx = argmax(W, axis=1)
        mat1 = repmat(idx.T, V.shape[0], 1)
        mat2 = repmat(idx, 1, V.shape[0])
        conn = elop(mat1, mat2, eq)

        return conn
