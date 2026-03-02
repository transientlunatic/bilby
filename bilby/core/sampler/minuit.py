import numpy as np
from pandas import DataFrame

from .base_sampler import Sampler, signal_wrapper


class Minuit(Sampler):
    """bilby wrapper of iminuit (https://iminuit.readthedocs.io/)

    Provides a frequentist maximum-likelihood interface using the MINUIT
    minimisation algorithm. Runs MIGRAD to find the best-fit parameters,
    optionally HESSE for Hessian-based covariance, and optionally MINOS for
    profile-likelihood confidence intervals.

    Unlike Bayesian samplers (e.g. dynesty, emcee), this sampler does *not*
    explore the full posterior distribution.  Instead it finds the
    maximum-likelihood estimate (MLE) and characterises the uncertainty around
    that point via a local Gaussian (HESSE covariance) or via profile
    likelihoods (MINOS).  The ``posterior`` stored in the bilby result is
    therefore a multivariate-Gaussian approximation centred on the MLE, not a
    true posterior.

    Frequentist results are stored in ``result.meta_data``:

    * ``best_fit``            – dict of MLE parameter values.
    * ``migrad_valid``        – whether MIGRAD converged.
    * ``migrad_accurate``     – whether MIGRAD is accurate (covariance valid).
    * ``migrad_fval``         – value of -2*log(L) at the minimum.
    * ``migrad_nfcn``         – number of likelihood evaluations used by MIGRAD.
    * ``hesse_covariance``    – covariance matrix (ndim × ndim numpy array).
    * ``hesse_errors``        – dict of symmetric 1-σ errors from HESSE.
    * ``minos_errors``        – dict of ``{param: (lower, upper)}`` asymmetric
                                1-σ errors from MINOS profile likelihood.
    * ``profiles``            – dict of ``{param: {'values': array,
                                'log_likelihood': array}}`` giving the
                                profile-likelihood curve for each parameter
                                (only present when ``compute_profiles=True``).

    Warm-starting from a previous bilby result
    -------------------------------------------
    If you have already run a Bayesian sampler (e.g. dynesty) and want to
    initialise MIGRAD near the posterior maximum, pass the path to the existing
    result file via ``start_from_result``::

        result = bilby.run_sampler(
            likelihood, priors, sampler="minuit",
            start_from_result="outdir/label_result.json",
        )

    The sample with the highest log-likelihood in the previous result is used
    as the starting point.  This is particularly effective when the Bayesian
    run has already located the likelihood peak, saving MIGRAD function calls.

    Parameters
    ==========
    nsamples : int
        Number of samples to draw from the Gaussian approximation around the
        maximum-likelihood estimate (default: 1000).
    run_hesse : bool
        If True (default), run HESSE after MIGRAD to obtain the Hessian-based
        covariance estimate.
    run_minos : bool
        If True (default: False), run MINOS after HESSE to compute
        profile-likelihood confidence intervals.  Requires ``run_hesse=True``.
    compute_profiles : bool
        If True (default: False), compute the profile-likelihood curve for
        the parameters listed in ``profile_parameters`` (or all search
        parameters when ``profile_parameters`` is None) and store the results
        in ``result.meta_data['profiles']``.  Requires ``run_hesse=True``.
    profile_parameters : list of str or None
        Subset of search parameter names for which to compute
        profile-likelihood curves when ``compute_profiles=True``.  If None
        (default), profiles are computed for every search parameter.  Use
        this to limit expensive profile computations to only the parameters of
        interest, e.g. ``profile_parameters=["chirp_mass", "luminosity_distance"]``.
    profile_size : int
        Number of scan points for each profile-likelihood curve when
        ``compute_profiles=True`` (default: 100).
    start_from_result : str or bilby.core.result.Result or None
        Path to a bilby result file (or a :class:`bilby.core.result.Result`
        object) whose highest-likelihood sample is used as the starting point
        for MIGRAD.  Useful for warm-starting from a previous Bayesian run
        (default: None, starting values are taken from the prior medians).
    migrad_ncall : int or None
        Maximum number of function calls for MIGRAD (default: None, iminuit
        chooses automatically).
    minos_ncall : int or None
        Maximum number of function calls for each MINOS interval
        (default: None).
    tol : float or None
        Tolerance passed to iminuit's ``tol`` attribute.  Controls the
        convergence criterion for MIGRAD (default: None, iminuit default).
    strategy : int
        iminuit strategy (0=fast, 1=default, 2=careful).  Default: 1.
    print_level : int
        Verbosity level for iminuit output (0=quiet, 1=normal, 2=verbose).
        Default: 0.

    """

    sampler_name = "minuit"

    default_kwargs = dict(
        nsamples=1000,
        run_hesse=True,
        run_minos=False,
        compute_profiles=False,
        profile_parameters=None,
        profile_size=100,
        start_from_result=None,
        migrad_ncall=None,
        minos_ncall=None,
        tol=None,
        strategy=1,
        print_level=0,
    )

    def _verify_external_sampler(self):
        try:
            import iminuit  # noqa: F401
        except ImportError:
            from .base_sampler import SamplerNotInstalledError

            raise SamplerNotInstalledError(
                "Sampler iminuit is not installed on this system. "
                "Install it via: pip install iminuit"
            )

    def _neg_log_likelihood(self, theta):
        """Return the negative log-likelihood for minimisation.

        Clips ``theta`` to the prior bounds before evaluating so that
        numerical steps taken by iminuit (during MIGRAD, HESSE, MINOS, or
        mnprofile) that land infinitesimally outside a hard bound (e.g.
        ``chi_2 = -1e-16``) do not raise errors in the waveform generator.
        """
        theta = np.array(theta, dtype=float)
        for i, key in enumerate(self._search_parameter_keys):
            lo, hi = self._prior_limits(key)
            if lo is not None:
                theta[i] = max(theta[i], lo)
            if hi is not None:
                theta[i] = min(theta[i], hi)
        return -self.log_likelihood(theta)

    @signal_wrapper
    def run_sampler(self):
        """Run iminuit to find the maximum-likelihood estimate.

        Returns
        =======
        bilby.core.result.Result
            Packaged information about the result.

        """
        import iminuit

        # Build the initial parameter values from the prior means/modes,
        # optionally warm-started from a previous bilby result.
        x0 = np.array(
            [
                self._initial_value(key)
                for key in self._search_parameter_keys
            ]
        )

        # Build parameter limits from prior bounds (None = no bound).
        limits = [
            self._prior_limits(key) for key in self._search_parameter_keys
        ]

        # Objective: negative log-likelihood accepting a numpy array.
        def fcn(x):
            return self._neg_log_likelihood(x)

        fcn.errordef = iminuit.Minuit.LIKELIHOOD

        m = iminuit.Minuit(fcn, x0, name=self._search_parameter_keys)
        m.limits = limits
        m.strategy = self.kwargs["strategy"]
        m.print_level = self.kwargs["print_level"]
        if self.kwargs["tol"] is not None:
            m.tol = self.kwargs["tol"]

        # ── MIGRAD ──────────────────────────────────────────────────────────
        migrad_ncall = self.kwargs["migrad_ncall"]
        m.migrad(ncall=migrad_ncall)

        if not m.valid:
            from ..utils import logger

            logger.warning(
                "MIGRAD did not converge to a valid minimum. "
                "Results may be unreliable."
            )

        # ── HESSE ───────────────────────────────────────────────────────────
        covariance = None
        if self.kwargs["run_hesse"]:
            m.hesse()
            if m.covariance is not None:
                covariance = np.array(m.covariance)

        # ── MINOS ───────────────────────────────────────────────────────────
        minos_errors = None
        if self.kwargs["run_minos"] and self.kwargs["run_hesse"]:
            if not m.valid:
                from ..utils import logger as _logger
                _logger.warning(
                    "Skipping MINOS because the function minimum is not valid "
                    "(MIGRAD did not converge). MINOS requires a valid minimum."
                )
            else:
                minos_ncall = self.kwargs["minos_ncall"]
                m.minos(ncall=minos_ncall)
                minos_errors = {
                    key: (m.merrors[key].lower, m.merrors[key].upper)
                    for key in self._search_parameter_keys
                    if key in m.merrors
                }

        # ── Profile likelihoods ──────────────────────────────────────────────
        profiles = None
        if self.kwargs["compute_profiles"] and self.kwargs["run_hesse"]:
            profiles = {}
            size = self.kwargs["profile_size"]
            profile_params = self.kwargs["profile_parameters"]
            if profile_params is None:
                profile_params = self._search_parameter_keys
            else:
                unknown = [p for p in profile_params if p not in self._search_parameter_keys]
                if unknown:
                    from ..utils import logger as _logger
                    _logger.warning(
                        f"profile_parameters contains keys not in search parameters: {unknown}. "
                        "They will be ignored."
                    )
                profile_params = [p for p in profile_params if p in self._search_parameter_keys]
            for key in profile_params:
                try:
                    x_vals, fvals, valid = m.mnprofile(key, size=size)
                    # Convert from FCN values to log-likelihood relative to max.
                    # FCN = -2 * log(L), so log(L) = -FCN/2.
                    # Profile log-likelihood ratio: Δlog(L) = -(fval - fmin)/2
                    logl = -(np.array(fvals) - m.fval) / 2
                    profiles[key] = {
                        "values": np.array(x_vals),
                        "log_likelihood": logl,
                    }
                except Exception:
                    pass

        # ── Collect best-fit values ──────────────────────────────────────────
        best_fit = {
            key: float(m.values[key])
            for key in self._search_parameter_keys
        }

        # ── Generate samples from Gaussian approximation ─────────────────────
        nsamples = self.kwargs["nsamples"]
        mean = np.array([m.values[k] for k in self._search_parameter_keys])

        if covariance is not None and np.all(np.isfinite(covariance)):
            try:
                samples = np.random.multivariate_normal(
                    mean, covariance, size=nsamples
                )
            except np.linalg.LinAlgError:
                # Fall back to diagonal when the full covariance is singular.
                # Use the prior width as the step size where available.
                from ..utils import logger as _logger

                _logger.warning(
                    "HESSE covariance matrix is singular; falling back to "
                    "diagonal covariance using prior widths."
                )
                diag = np.diag(covariance)
                for i, key in enumerate(self._search_parameter_keys):
                    if not (np.isfinite(diag[i]) and diag[i] > 0):
                        prior = self.priors[key]
                        lo = prior.minimum if np.isfinite(prior.minimum) else None
                        hi = prior.maximum if np.isfinite(prior.maximum) else None
                        if lo is not None and hi is not None:
                            diag[i] = ((hi - lo) / 6.0) ** 2
                        else:
                            diag[i] = 1.0
                samples = mean + np.random.randn(nsamples, len(mean)) * np.sqrt(diag)
        else:
            # No valid covariance: return only the best-fit point replicated.
            samples = np.tile(mean, (nsamples, 1))

        # Clip samples to prior bounds so that parameters with hard physical
        # limits (e.g. spin magnitudes in [0, 1]) are never passed out-of-range
        # to the likelihood.
        for i, key in enumerate(self._search_parameter_keys):
            lo, hi = self._prior_limits(key)
            samples[:, i] = np.clip(
                samples[:, i],
                lo if lo is not None else -np.inf,
                hi if hi is not None else np.inf,
            )

        self.result.samples = samples
        self.result.posterior = DataFrame(
            samples, columns=self._search_parameter_keys
        )
        self.result.log_likelihood_evaluations = np.array(
            [self.log_likelihood(s) for s in samples]
        )

        # ── Metadata ─────────────────────────────────────────────────────────
        self.result.log_evidence = np.nan
        self.result.log_evidence_err = np.nan

        self.result.meta_data["best_fit"] = best_fit
        self.result.meta_data["migrad_valid"] = m.valid
        self.result.meta_data["migrad_accurate"] = m.accurate
        self.result.meta_data["migrad_fval"] = float(m.fval)
        self.result.meta_data["migrad_nfcn"] = m.nfcn

        if covariance is not None:
            self.result.meta_data["hesse_covariance"] = covariance
            hesse_errors = {
                key: float(m.errors[key])
                for key in self._search_parameter_keys
            }
            self.result.meta_data["hesse_errors"] = hesse_errors

        if minos_errors is not None:
            self.result.meta_data["minos_errors"] = minos_errors

        if profiles is not None:
            self.result.meta_data["profiles"] = profiles

        self.calc_likelihood_count()
        return self.result

    def _run_test(self):
        """Run a quick test with a small number of function calls."""
        self.kwargs["migrad_ncall"] = 10
        self.kwargs["run_hesse"] = False
        self.kwargs["run_minos"] = False
        self.kwargs["compute_profiles"] = False
        self.kwargs["nsamples"] = 10
        return self.run_sampler()

    def write_current_state(self):
        """iminuit does not support checkpointing."""
        pass

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _initial_value(self, key):
        """Return a sensible starting value for parameter *key*.

        If ``start_from_result`` is set, the highest-likelihood sample from
        that result is used.  Otherwise, the prior median (``rescale(0.5)``)
        is used when finite, falling back to 0.
        """
        # Warm-start from a previously saved result.
        warm_start = self.kwargs.get("start_from_result")
        if warm_start is not None:
            val = self._warm_start_values.get(key)
            if val is not None and np.isfinite(val):
                return float(val)

        prior = self.priors[key]
        try:
            val = float(prior.rescale(0.5))
        except (ValueError, TypeError, AttributeError):
            val = 0.0
        if not np.isfinite(val):
            val = 0.0
        return val

    @property
    def _warm_start_values(self):
        """Dict of parameter → value from the warm-start result (cached)."""
        if not hasattr(self, "_warm_start_values_cache"):
            self._warm_start_values_cache = self._load_warm_start_values()
        return self._warm_start_values_cache

    def _load_warm_start_values(self):
        """Load the MAP sample from the warm-start result file or object."""
        from ..utils import logger as _logger

        warm_start = self.kwargs.get("start_from_result")
        if warm_start is None:
            return {}

        from ..result import read_in_result, Result

        if isinstance(warm_start, str):
            try:
                prior_result = read_in_result(filename=warm_start)
            except Exception as e:
                _logger.warning(
                    f"Could not load warm-start result from '{warm_start}': {e}. "
                    "Falling back to prior medians."
                )
                return {}
        elif isinstance(warm_start, Result):
            prior_result = warm_start
        else:
            _logger.warning(
                "start_from_result must be a file path string or a "
                "bilby.core.result.Result object. "
                "Falling back to prior medians."
            )
            return {}

        # Pick the highest-likelihood sample as the starting point.
        posterior = prior_result.posterior
        if "log_likelihood" in posterior.columns:
            map_row = posterior.iloc[posterior["log_likelihood"].idxmax()]
        else:
            _logger.warning(
                "Warm-start result has no 'log_likelihood' column; "
                "using the first sample as starting point."
            )
            map_row = posterior.iloc[0]

        _logger.info(
            "Warm-starting MIGRAD from maximum-likelihood sample in "
            f"'{warm_start}'."
        )
        return {
            k: float(map_row[k])
            for k in self._search_parameter_keys
            if k in map_row.index
        }

    def _prior_limits(self, key):
        """Return an (lower, upper) limit tuple for parameter *key*.

        Returns ``(None, None)`` when a bound is infinite.
        """
        prior = self.priors[key]
        lo = prior.minimum if np.isfinite(prior.minimum) else None
        hi = prior.maximum if np.isfinite(prior.maximum) else None
        return (lo, hi)
