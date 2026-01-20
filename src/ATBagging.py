import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from scipy.optimize import brenth
from dppy.finite_dpps import FiniteDPP
from dataclasses import dataclass
from tqdm import tqdm

def logpdet(A):
    '''
    Calculate the log of the pseudodeterminant of hermitian matrix A
    '''
    loglams = np.log(abs(np.linalg.eigvalsh(A)))
    logpdet = loglams[loglams>-15].sum()
    return logpdet

def _softplus_stable(s) -> float:
    if s > 50:
        return s
    if s < -50:
        return np.exp(s)
    return float(np.log1p(np.exp(s)))

@dataclass(frozen=True)
class DownselectResult:
    indices: np.ndarray          # selected points
    scores: np.ndarray | None    # informativeness scores

class ATBagging():
    def __init__(
            self, 
            base_estimator: BaseEstimator | None = None, 
            n_estimators: int = 100, 
            random_seed: int | None = None,
            n_jobs: int | None = None,
            model: BaggingRegressor | None = None
        ):
        '''
        BaggingRegressor + ATBagging Downselection

        Typical Usage
        -------------
        ```
        atb = ATBagging(n_estimators=100, random_seed=0)
        atb.fit(Xtr, ytr)
        downselection_results = atb.downselect(n=30, Xstar=Xstar)
        indices = downselection_results.indices
        ```
        '''
        # Initializations
        self.new_model_created = False
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.base_estimator = base_estimator \
                              if base_estimator is not None \
                              else DecisionTreeRegressor()
        self.model = model if model is not None else self._construct_model(n_jobs)
        self.raw_scores = None
        # Flags
        self.is_fit = False if self.new_model_created else True

    def _construct_model(self, n_jobs: int | None = None):
        '''
        Initialize the underlying BaggingRegressor model using the `base_estimator`
        
        Parameters
        ----------
        n_jobs : Optional, int
            Number of jobs for sklearn parallelization
            If None, -1 (all available cores) is used
        '''
        self.model = BaggingRegressor(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            bootstrap=True, 
            oob_score=True, 
            n_jobs=-1 if n_jobs is None else n_jobs,
            random_state=self.random_seed
        )
        self.new_model_created = True
        return self.model

    def fit(self, X: ArrayLike, y: ArrayLike):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model.fit(X, y)
        self.Xtr = X
        self.ytr = y
        self.n_train = len(X)
        self.is_fit = True
        return self

    def predict(self, X: ArrayLike):
        if not self.is_fit:
            raise ValueError("Bagging ensemble must be fit prior to prediction")
        return self.model.predict(np.array(X))

    def get_oob_samples(
            self,
            n: int | None = None,
            X: np.ndarray | None = None
        )->tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Returns `(oob_indices, inb_indices)`, two python lists each containing the 
        indices for in-bag and out-of-bag as ndarrays
        samples for each base estimator in the bagging ensemble.

        Parameters
        ----------
        n : Optional, int
            The number of data on which the BaggingRegressor was trained.
        X : Optional, ndarray 
            The data array on which the BaggingRegressor was trained.
            ! The content of X does not need to be the real data,
              only the length of the array is used.
       
        Raises
        ------
        ValueError
            When both arguments are None. At least one must be passed.
            (`n` supercedes `X.shape[0]` when both are passed)

        Note:
            This reimplements the logic from sklearn.ensemble._generate_indices
        '''
        if n is None and X is None:
            raise ValueError("At least one of n or X must be provided")
        if n is not None:
            n_samples = n
        elif X is not None:
            n_samples = X.shape[0]
        rng_state = check_random_state(self.random_seed)
        oob_indices = []
        inb_indices = []  # "in bag" -> inb, sue me
        all_indices = np.arange(n_samples)
        for _ in range(self.n_estimators):
            inds = rng_state.randint(0,n_samples,n_samples)
            inb_inds = np.unique(inds)
            inb_indices.append(inb_inds)
            oob_indices.append(np.delete(all_indices,inb_inds))
        return oob_indices, inb_indices

    def get_ensemble_predictions(
            self,
            X: np.ndarray,
        )->np.ndarray:
        '''
        Returns 2d ndarray of the predictions of each estimator of the ensemble,
        The first axis represents each estimator and the second the predictions
        '''
        if not self.is_fit:
            raise ValueError("Bagging Ensemble must be fit prior to prediction")
        preds = [est.predict(X) for est in self.model.estimators_]
        return np.asarray(preds, dtype=float).T

    def calculate_informativeness(
            self,
            Xstar: np.ndarray,
            eps: float = 1e-12,
            verbose: bool = False,
        )->np.ndarray:
        '''
        Calculate the informativeness score of each point in the training set.
        This is the KL divergence between the predictive distributions
        models whose training sets included and did not include the given point.

        Parameters
        ----------
        Xstar : ndarray
            The set of test points. The prediction distributions of the two model 
            classes (in-bag & out-of-bag) over this set are used for calculating the KL
            divergence & informativeness.
            This could be the training set, a subset of the training set, or any other
            example set tailored to the application.
        eps : float
            Ridge parameter for stability in covariance determinant calculations
        verbose : bool
            Print informativeness scores as they're calculated (spammy)

        Notes
        -----
        Worst case complexity is O(n_train×n_star**3), due to the cov+pinv per point
        Reduce the size of the probe set Xstar if time becomes an issue
        '''
        if not self.is_fit:
            raise ValueError(
                "Bagging Ensemble must be fit prior to informativeness calculations"
            )
        _, inb = self.get_oob_samples(n=self.n_train)
        all_inds = np.arange(self.n_train)
        inb_mask = np.array([np.isin(all_inds, i) for i in inb]).T
        preds_Xstar = self.get_ensemble_predictions(Xstar)
        KLs = np.zeros(self.n_train)
        for i in tqdm(range(self.n_train), 
                      disable=not verbose,
                      desc="Calculating informativeness scores",
                      total=self.n_train):
            mask_inb = inb_mask[i]
            mask_oob = ~mask_inb
            # if point is in-bag for all estimators, KL undefined -> set 0
            if mask_oob.sum()<2 or mask_inb.sum()<2:
                KLs[i] = 0.
                continue
            preds_inb = preds_Xstar[:,mask_inb]
            preds_oob = preds_Xstar[:,mask_oob]
            μ_inb     = preds_inb.mean(axis=1)
            μ_oob     = preds_oob.mean(axis=1)
            σ_inb     = np.cov(preds_inb, bias=False)
            σ_oob     = np.cov(preds_oob, bias=False)
            σ_inb     = σ_inb + eps*np.eye(σ_inb.shape[0])
            σ_oob     = σ_oob + eps*np.eye(σ_oob.shape[0])
            σ_oob_inv = np.linalg.pinv(σ_oob, hermitian=True)
            kl        = np.trace(σ_oob_inv@σ_inb)
            dμ        = (μ_oob - μ_inb)
            kl       += float(dμ@(σ_oob_inv@dμ))
            kl       -= σ_oob.shape[0]
            kl       += logpdet(σ_inb) - logpdet(σ_oob)
            KLs[i] = 0.5*kl
        self.raw_scores = KLs
        return KLs

    def _kl_to_q(self, KLs: np.ndarray, power: float = 2.):
        '''
        Transform informativeness scores via power transform
        '''
        q = KLs - np.min(KLs)
        q = np.maximum(q,0)
        return q**power

    def _random_fourier_features(
        self,
        X: np.ndarray,
        tau: float,
        n_fourier_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        '''
        Random Fourier Features for squared-exponential kernel
        Returns B with shape (n, R)

        Parameters
        ----------
        X : np.ndarray
            Data to transform
        tau : float
            Precision of normal random variables in RFFs, equivalent to SE Kernel λ
        n_fourier_features : int
            Dimensionality of RFF embedding
        rng : np.random.Generator
            Numpy random number generator
        '''
        w = rng.normal(0., 1./tau, size=(n_fourier_features, X.shape[1]))
        b = rng.uniform(0., 2.*np.pi, size=n_fourier_features)
        B = np.sqrt(2./n_fourier_features)*np.cos(X@w.T + b)
        return B

    def _dpp_sampler(
        self,
        q: np.ndarray,
        X: np.ndarray,
        tau: float,
        n: int,
        n_fourier_features: int = 2000,
        rng: np.random.Generator | None = None,
        max_brenth_iter: int = 20000,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Backend DPP creation & sampling.
        Builds an informativeness-modified L-ensemble DPP & samples from it via DPPy

        Parameters
        ----------
        q : ndarray
            (Transformed) informativeness scores
        X : ndarray
            Data set from which points are sampled (training set)
        tau : float 
            Length scale hyperparameter for data similarity kernel
            Controls how strong the repulsive effect of the DPP will be
        n : int
            Desired subset size
        n_fourier_features : int
            Number of random Fourier features to use in the approximation to the SE 
            kernel
        rng : Optional, numpy.random.Generator
            Numpy random number generator for use in importance sampling
        max_brenth_iter : int
            Maximum number of iterations to use in the internal Brent root finding step
        verbose : bool
            Print

        Returns
        -------
        indices : ndarray (k,)
            Indices of the selected samples 
        weights : ndarray (k,)  
            Informativeness scores
        '''
        if rng is None:
            rng = np.random.default_rng(self.random_seed)
        R_eff = int(max(n_fourier_features, 2*n))
        B = self._random_fourier_features(X, tau=tau, n_fourier_features=R_eff, rng=rng)
        B = B*q[:,None]
        evals, evecs = np.linalg.eigh(B.T@B)

        def expected_size(s)->float:
            # Root solve for s such that expected size matches n
            sp = _softplus_stable(s)
            λ = np.clip(evals, 0.0, None)
            return float((sp*λ/(1.+sp*λ)).sum())
        
        s = brenth(
            lambda s_: expected_size(s_) - n,
            -1e7, 1e70, 
            maxiter=max_brenth_iter
        )
        sp = _softplus_stable(s)

        λ = np.clip(evals, 0., None)
        denom = np.sqrt(np.maximum(sp*λ, 1e-300))
        L_evecs = np.nan_to_num((B@evecs)/denom, nan=0., posinf=0., neginf=0.)

        p = sp*λ/(1+sp*λ)
        sel_inds = np.where(rng.binomial(1, p))[0]
        if sel_inds.size == 0:
            sel_inds = np.argsort(-p)[:n]

        V = L_evecs[:,sel_inds]
        Q, _ = np.linalg.qr(V, mode='reduced')

        pdpp = FiniteDPP('correlation', K=Q@Q.T, projection=True)
        sample = np.array(pdpp.sample_exact())
        scores = np.array(pdpp.K).copy()
        if sample.size < n:
            marg = np.diag(scores)
            order = np.argsort(-marg)
            pad = [i for i in order if i not in set(sample)]
            sample = np.c_[sample, np.asarray(pad[:(n-sample.size)])].astype(int)
        elif sample.size > n:
            sample = sample[:n]

        scores = scores/max(scores.sum(), 1e-300)
        weights = scores[sample]/n
        if verbose:
            print(f"DPP sample size = {sample.size} (target {n})")
        return sample, weights

    def downselect(
        self,
        n: int,
        Xstar: np.ndarray | None = None,
        tau: float = 2.0,
        n_rand_fourier_features: int = 2000,
        q_power: float = 2.0,
        eps: float = 1e-12,
        verbose: bool = False,
        rng: np.random.Generator | None = None,
    ):
        '''
        Sample n informative & diverse points from the training set.
        Calculates the informativeness for each point & a feature-space correlation 
        matrix, then uses this to parameterize a DPP, from which the set is sampled.

        Parameters
        ----------
        n : int
            Desired subset size
        Xstar : Optional, ndarray
            Probe points over which to measure prediction distribution shifts.
            If None, defaults to Xtr (common but expensive if Xtr is large).
        tau : float
            Lengthscale parameter in similarity kernel
        rff_R : int
            Number of random Fourier features to use in kernel approximation
        q_power : float
            Power used in informativeness score transformation
        eps : float
            Ridge parameter for the stability of determinant calculations
        verbose : bool
            Toggle printing intermediate information for debugging & tracking purposes
        rng : Optional, numpy.random.Generator
            Numpy random number generator, if None one is created and seeded with the 
            class's `random_seed` attribute
        '''
        if not self.is_fit:
            raise ValueError('Bagging Ensemble must be fit prior to downselection')
        Xtr = self.Xtr
        if Xstar is None:
            Xstar = Xtr
        else:
            Xstar = np.asarray(Xstar)
        if n<=0 or n>len(Xtr):
            raise ValueError(f"n must be in [1, n_train]; got n={n}, n_train={len(Xtr)}")
        KLs = self.calculate_informativeness(Xstar, eps=eps, verbose=verbose)
        q = self._kl_to_q(KLs, power=q_power)
        if not np.isfinite(q).all() or q.sum()<=0:
            q = np.ones_like(q)
        q = q/q.sum()
        inds, wts = self._dpp_sampler(
            q=q,
            X=Xtr,
            tau=tau,
            n=n,
            n_fourier_features=n_rand_fourier_features,
            rng=rng,
            verbose=verbose,
        )
        return DownselectResult(
            indices=inds,
            scores=wts,
        )

    

