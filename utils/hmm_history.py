"""
Credit for code goes to Dr. Christian Donner of the SDSC (https://datascience.ch/team_member/christian-donner/)
"""


import torch
import numpy as np
from typing import Iterable, Tuple

dtype = torch.double


class GaussianARHMM:
    def __init__(
        self, X: np.ndarray, K: int, H: int, noise: float = 1e-4, seed: int = None
    ):
        """Initializes HMM with autoregressive observation model, i.e. we have an latent integer variable

            z_t ~ P(z|z_{t-1}) with z in {0,K-1},

            where P(z=k|z_{t-1}=l) = T_{k,l} is a transition matrix, such that \sum_k T_{k,l} = 1.
            If z_t = k the observation x_t is coming from a Gaussian

            x_t ~ N(\mu_k(x_{t-H:t-1}), \Sigma_k),

            where the mean is a linear function of the data history

            \mu_k(x_{t-H:t-1}) = F_k x_{t-H:t-1} + b_k.

            Note, that \mu_k and \Sigma_k are different for each class k.

        :param X: Data matrix [timesteps x obs. dimensions]
        :type X: np.ndarray
        :param K: Number of states, the model can assume.
        :type K: int
        :param H: History length that is considered.
        :type H: int
        :param noise: Small noise parameter for numerical stability, defaults to 1e-4
        :type noise: float, optional
        :param seed: Random seed, defaults to None
        :type seed: int, optional
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.K = K
        self.X = torch.tensor(X, dtype=dtype)
        self.T, self.D = self.X.shape
        self.H = H
        self.X_history = self._construct_history()

        # Setup the parameters for each state.
        self.param_dict = {}
        for k in range(self.K):
            Sigma = torch.eye(self.D, dtype=dtype)
            L = torch.linalg.cholesky(Sigma)
            logdet_Sigma = 2.0 * torch.sum(torch.log(torch.diag(L)))
            Sigma_i = torch.cholesky_inverse(L)
            F = torch.zeros((self.D * self.H, self.D), dtype=dtype)
            F[: self.D] = torch.eye(self.D, dtype=dtype)
            b = 1e-4 * torch.randn(1, self.D, dtype=dtype)
            self.param_dict[k] = {
                "Sigma": Sigma,
                "L": L,
                "logdet_Sigma": logdet_Sigma,
                "Sigma_i": Sigma_i,
                "F": F,
                "b": b,
            }

        # Transition matrix
        self.p_trans = 0.9 * torch.eye(self.K, dtype=dtype) + 0.1
        self.p_trans /= torch.sum(self.p_trans, axis=0)
        # Initial state probabilities (Uniform)
        self.p_z0 = torch.ones(self.K) / self.K
        self.Q_list = []
        self.noise = noise

    def delete_data(self):
        """Deletes the data in the model."""
        self.X = None
        self.X_history = None

    def _construct_history(self, X: np.ndarray = None) -> torch.Tensor:
        """Constructs the history matrix.

        :param X: Data matrix [timesteps x obs. dimensions]. If not provided the one from the instance is taken, defaults to None
        :type X: np.ndarray, optional
        :return: History matrix [timesteps - H x obs. dimensions], where H times the data is copied shifted by 1 timestep.
        :rtype: torch.Tensor
        """
        if X is None:
            X = self.X
        T = X.shape[0]
        X_history = torch.empty((T - self.H, self.H * self.D), dtype=dtype)
        for h in range(1, self.H + 1):
            X_history[:, (h - 1) * self.D : h * self.D] = X[self.H - h : -h]
        return X_history

    def run_em(self, conv: float = 1e-4, min_iter: int = 20) -> torch.Tensor:
        """Runs the expectation maximization (EM) algorithm to fit the model.

        :param conv: Convergence criterion, defaults to 1e-4
        :type conv: float, optional
        :param min_iter: Min. number of EM iterations, defaults to 20
        :type min_iter: int, optional
        :return: Probabilities of state at a certain time point [timesteps x K]
        :rtype: torch.Tensor
        """
        qZ, qZZ, logPx = self.estep()
        self.Q_list.append(self.compute_Qfunc(qZ, qZZ, logPx))
        iteration = 0
        conv_crit = conv
        while conv_crit >= conv or iteration < min_iter:
            update_b = (iteration % 2) == 0
            self.mstep(qZ, qZZ, update_b)
            qZ, qZZ, logPx = self.estep()
            self.Q_list.append(self.compute_Qfunc(qZ, qZZ, logPx))
            iteration += 1
            conv_crit = torch.absolute(
                self.Q_list[-1] - self.Q_list[-2]
            ) / torch.absolute(
                torch.max(torch.stack([self.Q_list[-1], self.Q_list[-2]]))
            )
            print("Iteration %d - conv. crit: %.4f > %.4f" %(iteration, conv_crit, conv))
        return qZ

    def compute_Qfunc(
        self, qZ: torch.Tensor, qZZ: torch.Tensor, logPx: torch.Tensor
    ) -> float:
        """Computes the Q function

         Q(param) = E[ln p(X|Z, param)] + E[ln q(Z| param)]

        Expectations are over the trajectory of latent variables Z=(z_1,...,z_T)

        :param qZ: Probabilities of state at a certain time point [timesteps x K]
        :type qZ: torch.Tensor
        :param qZZ: Pairwise probabilities of state pairs at two consecutive time points [timesteps-1 x K x K]
        :type qZZ: torch.Tensor
        :param logPx: Log likelihoods for each state [timesteps x K]
        :type logPx: torch.Tensor
        :return: Q-function.
        :rtype: float
        """
        # likelihood term
        Q = torch.sum(qZ[1:] * logPx) + torch.sum(
            torch.sum(qZZ, axis=0) * torch.log(self.p_trans)
        )
        # entropy term
        Hqzz = qZZ[qZZ > 0] * torch.log(qZZ[qZZ > 0])
        Hqz = qZ[1:-1][qZ[1:-1] > 0] * torch.log(qZ[1:-1][qZ[1:-1] > 0])
        Q += -Hqzz[~torch.isnan(Hqzz)].sum() + Hqz[~torch.isnan(Hqz)].sum()
        return Q

    def get_data_log_likelihoods_class(
        self, k: int, X: np.ndarray = None
    ) -> torch.Tensor:
        """Computes the log likelihoods for a given class

            ln p(x_t|x_{t-H:t-1}, z_t=k).

        :param k: Class index.
        :type k: int
        :param X: Data [timestepy x obs. dim], defaults to None
        :type X: np.ndarray, optional
        :return: Log likelihoods for class k [timesteps - H]
        :rtype: torch.Tensor
        """
        # locals().update(self.param_dict[k])
        F, b, Sigma_i, logdet_Sigma = (
            self.param_dict[k]["F"],
            self.param_dict[k]["b"],
            self.param_dict[k]["Sigma_i"],
            self.param_dict[k]["logdet_Sigma"],
        )
        if X is None:
            X = self.X
            X_history = self.X_history
        else:
            X_history = self._construct_history(X)
        mu_t = torch.mm(X_history, F) + b
        dX = X[self.H :] - mu_t
        logPx_k = -0.5 * (
            torch.sum(torch.mm(dX, Sigma_i) * dX, axis=1)
            + logdet_Sigma
            + self.D * torch.log(2.0 * torch.tensor(np.pi, dtype=dtype))
        )
        return logPx_k

    def get_data_log_likelihoods(self, X: np.ndarray = None) -> torch.Tensor:
        """Gets the data log likelihood for all classes.

        :param X: Data [timesteps x obs. dim], defaults to None
        :type X: np.ndarray, optional
        :return: Log likelihoods for class k [timesteps - H x K]
        :rtype: torch.Tensor
        """
        if X is None:
            T = self.T
        else:
            T = X.shape[0]
        logPx = torch.empty([T - self.H, self.K], dtype=dtype)
        for k in range(self.K):
            logPx[:, k] = self.get_data_log_likelihoods_class(k, X=X)
        return logPx

    def compute_loglikelihood(self, X: np.ndarray = None) -> float:
        """Computes the log likelihood of a data point (given all classes).

        :param X: Data [timesteps x obs. dim], defaults to None
        :type X: np.ndarray, optional
        :return: Log likelihood of the data given the model.
        :rtype: torch.Tensor
        """
        logPx = self.get_data_log_likelihoods(X=X)
        fwd_messages, log_pXt = self.forward_pass(logPx)
        llk = torch.sum(log_pXt)
        return llk

    def get_num_params(self) -> int:
        """Returns number of free parameters in the model.

        :return: Number of free parameters.
        :rtype: int
        """
        num_params_per_class = (
            self.D**2 * self.H + self.D + self.D * (self.D - 1) / 2 + self.D
        )
        num_params = (self.K - 1) + self.K * (self.K - 1) + self.K * num_params_per_class
        return num_params



    def compute_AIC(self, X: np.ndarray = None) -> float:
        """Computes the Akaike information criterion for given data.

        :param X: _description_, defaults to None
        :type X: np.ndarray, optional
        :return: Akaike information criterion.
        :rtype: float
        """
        llk = self.compute_loglikelihood(X)
        num_params = self.get_num_params()
        aic = 2 * num_params - 2 * llk
        return aic
    
    
    def compute_BIC(self, X: np.ndarray = None) -> float:
        """Computes the Bayesian information criterion for given data.

        :param X: _description_, defaults to None
        :type X: np.ndarray, optional
        :return: Bayesian information criterion.
        :rtype: float
        """
        n = X.shape[0]
        llk = self.compute_loglikelihood(X)
        num_params = self.get_num_params()
        bic = np.log(self.T) * num_params - np.log(n) * llk
        return bic



    def estep(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the e-step, i.e. does the forward-backward pass over the data.

        :return: Returns the marginal, pairwise probabilities of the latent state, and the log likelihoods.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        logPx = self.get_data_log_likelihoods()
        fwd_messages, log_pXt = self.forward_pass(logPx)
        bwd_messages = self.backward_pass(logPx)
        qZ, qZZ = self.compute_marginals(fwd_messages, bwd_messages, logPx)
        return qZ, qZZ, logPx

    def forward_pass(self, logPx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Does the forward pass

            p(z_t|x_{1:t}) \propto p(x_t|z_t)\sum_{z_{t-1}} p(z_t|z_{t-1})p(z_{t-1}|x_{1:t-1})

        :param logPx: Log likelihoods for data and states. [T, K]
        :type logPx: torch.Tensor
        :return: The forward messages p(z_t|x_{1:t}), and the log likelihoods ln p(x_t|x_{1:t-1}).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        T = logPx.shape[0]
        fwd_messages = torch.empty([T + 1, self.K], dtype=dtype)
        log_pXt = torch.zeros([T + 1], dtype=dtype)
        fwd_messages[0] = self.p_z0
        for t in range(1, T + 1):
            log_msg = logPx[t - 1 : t] + torch.log(
                torch.mm(fwd_messages[t - 1 : t], self.p_trans.T)
            )
            log_pXt[t] = torch.logsumexp(log_msg, 1)
            fwd_messages[t] = torch.exp(log_msg - log_pXt[t])
        return fwd_messages, log_pXt

    def backward_pass(self, logPx: torch.Tensor) -> torch.Tensor:
        """Does the backward pass.

        :param logPx: Log likelihoods for data and states. [T, K]
        :type logPx: torch.Tensor
        :return: Return backward messages p(z_t|x_{t+1:T})
        :rtype: torch.Tensor
        """
        T = logPx.shape[0]
        bwd_messages = torch.empty([T + 1, self.K], dtype=dtype)
        bwd_messages[-1] = 1.0

        for t in range(T - 1, -1, -1):
            if t > 0:
                log_msg = logPx[t - 1] + torch.log(
                    torch.mm(bwd_messages[t + 1 : t + 2], self.p_trans)
                )
            else:
                log_msg = torch.log(torch.mm(bwd_messages[t + 1 : t + 2], self.p_trans))
            lognorm = torch.logsumexp(log_msg, 1)
            bwd_messages[t] = torch.exp(log_msg - lognorm)
        return bwd_messages

    def compute_marginals(
        self,
        fwd_messages: torch.Tensor,
        bwd_messages: torch.Tensor,
        logPx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the marginals p(z_t|x_{1:T}), and pairwise marginals p(z_t,z_{t+1}|x_{1:T}) from the forward and backward messages.

        :param fwd_messages: Forward messages [T, K]
        :type fwd_messages: torch.Tensor
        :param bwd_messages: Backward messages [T, K]
        :type bwd_messages: torch.Tensor
        :param logPx: Observation log likelihoods given a state [T, K]
        :type logPx: torch.Tensor
        :return: Marginals, and pairwise marginals. [T,K], [T-1,K,K]
        :rtype: Tuple[ torch.Tensor, torch.Tensor]
        """
        log_qZ = torch.log(fwd_messages[:]) + torch.log(bwd_messages[:])
        log_qZ[1:] -= logPx
        logZ = torch.logsumexp(log_qZ, axis=1)
        qZ = torch.exp(log_qZ - logZ[:, None])

        log_qZZ = (
            torch.log(fwd_messages[:-1, :, None])
            + torch.log(bwd_messages[1:, None])
            + torch.log(self.p_trans.T)
        )
        logZZ = torch.logsumexp(torch.logsumexp(log_qZZ, axis=1), axis=1)
        qZZ = torch.exp(log_qZZ - logZZ[:, None, None])
        return qZ, qZZ

    def mstep(self, qZ: torch.Tensor, qZZ: torch.Tensor, update_b: bool):
        """Updates the model parameters analytically.

        :param qZ: Marginal state probability [T, K]
        :type qZ: torch.Tensor
        :param qZZ: Pairwise marginal state probabilities. [T-1, K, K]
        :type qZZ: torch.Tensor
        :param update_b: Whether bias in the observatin should be updated.
        :type update_b: bool
        """
        self.update_state_transitions(qZ, qZZ)
        self.update_state_params(qZ, qZZ, update_b)

    def update_state_transitions(self, qZ: torch.Tensor, qZZ: torch.Tensor):
        """Updates the transition matrix.

        :param qZ: Marginal state probability [T, K]
        :type qZ: torch.Tensor
        :param qZZ: Pairwise marginal state probabilities. [T-1, K, K]
        :type qZZ: torch.Tensor
        """
        self.p_trans = torch.sum(qZZ, axis=0) + 1e-8
        self.p_trans /= torch.sum(self.p_trans, axis=0)

    def update_state_params(
        self, qZ: torch.Tensor, qZZ: torch.Tensor, update_b: bool = True
    ):
        """Updates the state observation models.

        :param qZ: Marginal state probability [T, K]
        :type qZ: torch.Tensor
        :param qZZ: Pairwise marginal state probabilities. [T-1, K, K]
        :type qZZ: torch.Tensor
        :param update_b: Whether the bias should be updated, defaults to True
        :type update_b: bool, optional
        """
        sum_qZ = torch.sum(qZ[1:], axis=0)
        for k in range(self.K):
            # read out parameters
            # locals().update(self.param_dict[k])
            F, b = self.param_dict[k]["F"], self.param_dict[k]["b"]
            # calculate optimal Sigma
            mu_t = torch.mm(self.X_history, F) + b
            dX = self.X[self.H :] - mu_t
            dX_weighted = qZ[1:, k : k + 1] * dX
            Sigma_new = torch.mm(dX_weighted.T, dX) / sum_qZ[
                k
            ] + self.noise * torch.eye(self.D, dtype=dtype)
            L_new = torch.linalg.cholesky(Sigma_new)
            logdet_Sigma_new = 2.0 * torch.sum(torch.log(torch.diag(L_new)))
            Sigma_i_new = torch.cholesky_inverse(L_new)
            # calculate optimal beta
            if update_b:
                dX = self.X[self.H :] - torch.mm(self.X_history, F)
                b_new = torch.mm(qZ[1:, k : k + 1].T, dX) / sum_qZ[k]
                F_new = F
            else:
                # calculate optimal F
                dX = qZ[1:, k : k + 1] * (self.X[self.H :] - b)
                A = torch.mm(dX.T, self.X_history)
                B = torch.inverse(
                    torch.mm(self.X_history.T, qZ[1:, k : k + 1] * self.X_history)
                    + self.noise * torch.eye(self.X_history.shape[1])
                )
                F_new = torch.mm(A, B).T
                b_new = b
            self.param_dict[k] = {
                "Sigma": Sigma_new,
                "L": L_new,
                "logdet_Sigma": logdet_Sigma_new,
                "Sigma_i": Sigma_i_new,
                "F": F_new,
                "b": b_new,
            }

    def set_observed_vars(self, observed_idx: Iterable):
        """Sets the set of observed variables. (Only for prediction).

        :param observed_idx: Indices of observed variables.
        :type observed_idx: Iterable
        """
        self.oidx = observed_idx
        self.num_observed = len(self.oidx)
        self.oidx_hist = np.concatenate([self.oidx + self.D * h for h in range(self.H)])
        ones_arr = np.ones(self.D)
        ones_arr[self.oidx] = 0
        self.uidx = np.where(ones_arr)[0]
        self.num_unobserved = len(self.uidx)
        self.uidx_hist = np.concatenate([self.uidx + self.D * h for h in range(self.H)])
        mask1_u = np.zeros([self.D, self.D])
        mask1_u[self.uidx] = 1
        mask2_u = np.zeros([self.D, self.D])
        mask2_u[:, self.uidx] = 1
        self.uuidx = np.where(mask1_u * mask2_u)
        mask1_o = np.zeros([self.D, self.D])
        mask1_o[self.oidx] = 1
        mask2_o = np.zeros([self.D, self.D])
        mask2_o[:, self.oidx] = 1
        self.ooidx = np.where(mask1_o * mask2_o)
        self.ouidx = np.where(mask1_o * mask2_u)

    def prediction(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Does the prediction of variables, given data. Predictions are conditioned on observed indices. To have analytical predictions moment matching is used.

        :param X: Date
        :type X: torch.Tensor
        :return: (Filter) state probabilities, mean, and variance of the unsobserved data.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        X_history = self._construct_history(X)
        T = X.shape[0]  # 10000
        pz_val = torch.ones((T - self.H + 1, self.K), dtype=dtype) / self.K
        mu_u_val = torch.zeros(
            (T - self.H + 1, self.num_unobserved * self.H), dtype=dtype
        )
        Sigma_uu_val = torch.zeros(
            (
                T - self.H + 1,
                self.num_unobserved * self.H,
                self.num_unobserved * self.H,
            ),
            dtype=dtype,
        )
        Sigma_uu_val[0] = torch.eye(self.num_unobserved * self.H, dtype=dtype)

        for t_idx in range(1, T - self.H):
            pz_past = pz_val[t_idx - 1 : t_idx]
            mu_u_past = mu_u_val[t_idx - 1 : t_idx]
            Sigma_uu_past = Sigma_uu_val[t_idx - 1]
            x_o_cur, x_o_past = (
                X[t_idx + self.H - 1 : t_idx + 1 + self.H - 1, self.oidx],
                X_history[t_idx - 1 : t_idx, self.oidx_hist],
            )
            # prediction z_t
            pz_pred = torch.mm(pz_past, self.p_trans.T)
            # prediction x_t
            logp_pred_x_o = torch.empty((1, self.K), dtype=dtype)
            mu_u_filt, Exx_u_filt = torch.zeros(
                (self.K, self.H * self.num_unobserved), dtype=dtype
            ), torch.zeros(
                (self.K, self.H * self.num_unobserved, self.H * self.num_unobserved),
                dtype=dtype,
            )
            for k in range(self.K):
                # ADF representation for x_t
                mu_u_filt[k], Exx_u_filt[k], logp_pred_x_o[:, k] = self.filtering(
                    k, mu_u_past, Sigma_uu_past, x_o_cur, x_o_past, pz_pred
                )
            logpx_o_t_filt = torch.logsumexp(logp_pred_x_o, 1)
            pz_filt = torch.exp(logp_pred_x_o - logpx_o_t_filt)
            mu_u_filt = torch.mm(pz_filt, mu_u_filt)
            Exx = torch.sum((pz_filt * Exx_u_filt.T), axis=2)
            Sigma_uu_filt = Exx - torch.mm(mu_u_filt.T, mu_u_filt)
            # makes it numerically stable
            Sigma_uu_filt = 0.5 * (Sigma_uu_filt + Sigma_uu_filt.T)
            pz_val[t_idx : t_idx + 1] = pz_filt
            mu_u_val[t_idx : t_idx + 1] = mu_u_filt
            Sigma_uu_val[t_idx] = Sigma_uu_filt
        return pz_val, mu_u_val, Sigma_uu_val

    def filtering(
        self,
        k: int,
        mu_u_past: torch.Tensor,
        Sigma_uu_past: torch.Tensor,
        x_o_cur: torch.Tensor,
        x_o_past: torch.Tensor,
        pz_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filtering via moment matching.

        :param k: State index
        :type k: int
        :param mu_u_past: Mean of the unobserved variabels at t-1.
        :type mu_u_past: torch.Tensor
        :param Sigma_uu_past: Covariance of the unobserved variabels at t-1.
        :type Sigma_uu_past: torch.Tensor
        :param x_o_cur: Observed data at t.
        :type x_o_cur: torch.Tensor
        :param x_o_past: Previviously observed data.
        :type x_o_past: torch.Tensor
        :param pz_pred: Prediction of latent state, i.e. p(z_t|x_{1:t-1})
        :type pz_pred: torch.Tensor
        :return: Mean, and second order moment for the unobserved, and log likelihood for the observed variables.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        F, b, Sigma = (
            self.param_dict[k]["F"],
            self.param_dict[k]["b"],
            self.param_dict[k]["Sigma"],
        )
        Fu, Fo = F[self.uidx_hist], F[self.oidx_hist]
        mu_pred = torch.mm(mu_u_past, Fu) + torch.mm(x_o_past, Fo) + b
        Sigma_pred = Sigma + torch.mm(Fu.T, torch.mm(Sigma_uu_past, Fu))
        # torch.linalg.cholesky(Sigma_pred)
        # marginal value for x_o_cur
        Sigma_pred_oo = Sigma_pred[self.ooidx].reshape(
            self.num_observed, self.num_observed
        )
        Sigma_pred_uu = torch.empty(
            (self.H * self.num_unobserved, self.H * self.num_unobserved), dtype=dtype
        )
        Sigma_pred_uu[: self.num_unobserved, : self.num_unobserved] = Sigma_pred[
            self.uuidx
        ].reshape(self.num_unobserved, self.num_unobserved)
        Sigma_pred_uu[self.num_unobserved :, self.num_unobserved :] = Sigma_uu_past[
            : -self.num_unobserved, : -self.num_unobserved
        ]
        Sigma_pred_uu[self.num_unobserved :, : self.num_unobserved] = torch.mm(
            Sigma_uu_past[: -self.num_unobserved], Fu[:, self.uidx]
        )
        Sigma_pred_uu[: self.num_unobserved, self.num_unobserved :] = Sigma_pred_uu[
            self.num_unobserved :, : self.num_unobserved
        ].T
        L_pred_oo = torch.linalg.cholesky(
            Sigma_pred_oo
        )  # this can be done more efficiently (?)
        Sigma_pred_oo_i = torch.cholesky_inverse(L_pred_oo)
        logdet_Sigma_pred_oo = 2.0 * torch.sum(torch.log(L_pred_oo.diag()))
        dx_o = x_o_cur - mu_pred[:, self.oidx]
        log_weight = torch.log(pz_pred[:, k])
        # ADF representation for x_t
        Sigma_hat_off_diag = torch.mm(
            Sigma_uu_past[: -self.num_unobserved], Fu[:, self.oidx]
        )
        Sigma_ou = torch.empty(
            (self.num_observed, self.num_unobserved * self.H), dtype=dtype
        )
        Sigma_ou[:, : self.num_unobserved] = Sigma_pred[self.ouidx].reshape(
            (self.num_observed, self.num_unobserved)
        )
        Sigma_ou[:, self.num_unobserved :] = Sigma_hat_off_diag.T
        mu_u_filt = torch.empty((1, self.H * self.num_unobserved), dtype=dtype)
        mu_u_filt[:, : self.num_unobserved] = mu_pred[:, self.uidx]
        mu_u_filt[:, self.num_unobserved :] = mu_u_past[:, : -self.num_unobserved]
        mu_u_filt += torch.mm(
            torch.mm((x_o_cur - mu_pred[:, self.oidx]), Sigma_pred_oo_i), Sigma_ou
        )
        # torch.linalg.cholesky(Sigma_pred_uu)
        Sigma_uu_filt = Sigma_pred_uu - torch.mm(
            Sigma_ou.T, torch.mm(Sigma_pred_oo_i, Sigma_ou)
        )
        # torch.linalg.cholesky(Sigma_uu_filt_k)
        # weight
        logp_pred_x_o = log_weight - 0.5 * (
            torch.sum(torch.mm(dx_o, Sigma_pred_oo_i) * dx_o, axis=1)
            + logdet_Sigma_pred_oo
            + self.num_observed * torch.log(2.0 * torch.tensor(np.pi, dtype=dtype))
        )
        Exx_u_filt = Sigma_uu_filt + torch.mm(mu_u_filt.T, mu_u_filt)
        return mu_u_filt, Exx_u_filt, logp_pred_x_o

    def sample_trajectory(
        self, T: int, x0: torch.Tensor, z0: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples a trajectory.

        :param T: Number of timesteps to be sampled
        :type T: int
        :param x0: Starting observation
        :type x0: torch.Tensor
        :param z0: Initial state
        :type z0: int
        :return: Data, and latent state trajectory.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        z = torch.empty([T + 1], dtype=dtype)
        z[0] = z0
        rand_num_z = torch.rand(T)
        x = torch.empty([T + 1, self.D], dtype=dtype)
        x[0] = x0
        rand_num_x = torch.randn((T, self.D))
        for t in range(T):
            cum_p_z = torch.cumsum(self.p_trans[:, z[t]], 0)
            z[t + 1] = torch.searchsorted(cum_p_z, rand_num_z[0])
            F, b, L = (
                self.param_dict[z[t + 1]]["F"],
                self.param_dict[z[t + 1]]["b"],
                self.param_dict[z[t + 1]]["L"],
            )
            mu_t = torch.mm(x[t : t + 1], F) + b
            x[t] = mu_t + torch.mm(L, rand_num_x[t])
        return x, z

    def sample_conditional_trajectory(
        self, X: torch.Tensor, z0: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples trajectory conditioned on observed data.

        :param X: Data.
        :type X: torch.Tensor
        :param z0: Initial state.
        :type z0: int
        :return: Data, and latent state trajectory.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        T = X.shape[0]
        z = torch.empty([T - self.H + 1], dtype=int)
        z[0] = z0
        rand_num_z = torch.rand(T, dtype=dtype)
        rand_num_x = torch.randn((T, self.num_unobserved), dtype=dtype)
        Sigma_oo_list = []
        Sigma_oo_i_list = []
        L_oo_list = []
        for k in range(self.K):
            Sigma = self.param_dict[k]["Sigma"]
            Sigma_oo = Sigma[self.ooidx].reshape(self.num_observed, self.num_observed)
            Sigma_oo = 0.5 * (Sigma_oo + Sigma_oo.T)
            Sigma_oo_list.append(Sigma_oo)
            L_oo = torch.linalg.cholesky(Sigma_oo)
            L_oo_list.append(L_oo)  # this can be done more efficiently (?)
            Sigma_oo_i_list.append(torch.cholesky_inverse(L_oo))

        for t_idx in range(self.H, T):
            z_past = int(z[t_idx - self.H])
            x_o_cur = X[t_idx : t_idx + 1, self.oidx]
            x_history = torch.flip(X[t_idx - self.H : t_idx], (0,)).reshape(
                (1, self.D * self.H)
            )
            pz_pred = self.p_trans[:, z_past]
            logp_x_o = torch.empty((1, self.K), dtype=dtype)
            for k in range(self.K):
                log_weight = torch.log(pz_pred[k])
                F, b, Sigma = (
                    self.param_dict[k]["F"],
                    self.param_dict[k]["b"],
                    self.param_dict[k]["Sigma"],
                )
                # Sigma_oo = Sigma[self.ooidx].reshape(self.num_observed, self.num_observed)
                # Sigma_oo = .5 * (Sigma_oo + Sigma_oo.T)
                # L_oo = torch.linalg.cholesky(Sigma_oo) # this can be done more efficiently (?)
                # Sigma_oo_i = torch.cholesky_inverse(L_oo)
                Sigma_oo = Sigma_oo_list[k]
                Sigma_oo_i = Sigma_oo_i_list[k]
                L_oo = L_oo_list[k]
                logdet_Sigma_oo = 2.0 * torch.sum(torch.log(L_oo.diag()))
                mu = torch.mm(x_history, F) + b
                dx_o = x_o_cur - mu[:, self.oidx]
                logp_x_o[0, k] = log_weight - 0.5 * (
                    torch.sum(torch.mm(dx_o, Sigma_oo_i) * dx_o, axis=1)
                    + logdet_Sigma_oo
                    + self.num_observed
                    * torch.log(2.0 * torch.tensor(np.pi, dtype=dtype))
                )
            logpx_o_t = torch.logsumexp(logp_x_o, 1)
            pz = torch.exp(logp_x_o - logpx_o_t)[0]
            cum_pz = torch.cumsum(pz, 0)
            z[t_idx - self.H + 1] = torch.searchsorted(cum_pz, rand_num_z[t_idx])
            z_cur = int(z[t_idx - self.H])
            F, b, Sigma = (
                self.param_dict[z_cur]["F"],
                self.param_dict[z_cur]["b"],
                self.param_dict[z_cur]["Sigma"],
            )
            mu = torch.mm(x_history, F) + b
            # Sigma_oo = Sigma[self.ooidx].reshape(self.num_observed, self.num_observed)
            # L_oo = torch.linalg.cholesky(Sigma_oo) # this can be done more efficiently (?)
            # Sigma_oo_i = torch.cholesky_inverse(L_oo)
            Sigma_oo = Sigma_oo_list[z_cur]
            Sigma_oo_i = Sigma_oo_i_list[z_cur]
            L_oo = L_oo_list[z_cur]
            Sigma_ou = Sigma[self.ouidx].reshape(
                (self.num_observed, self.num_unobserved)
            )
            Sigma_uu = Sigma[self.uuidx].reshape(
                self.num_unobserved, self.num_unobserved
            )
            mu_u = mu[:, self.uidx] + torch.mm(
                torch.mm((x_o_cur - mu[:, self.oidx]), Sigma_oo_i), Sigma_ou
            )
            Sigma_uu = Sigma_uu - torch.mm(Sigma_ou.T, torch.mm(Sigma_oo_i, Sigma_ou))
            Sigma_uu = 0.5 * (Sigma_uu + Sigma_uu.T)
            L_uu = torch.linalg.cholesky(Sigma_uu)
            X[t_idx : t_idx + 1, self.uidx] = (
                mu_u + torch.mm(L_uu, rand_num_x[t_idx : t_idx + 1].T).T
            )
        return X, z


# A_non_idx = A[np.where(mask_not)].reshape(N-len(idx), N-len(idx))
# A_off_diag =  A[np.where(mask_cross)].reshape(N-len(idx), len(idx))
