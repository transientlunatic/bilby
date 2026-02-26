import numpy as np
from pandas import DataFrame

from .base_sampler import Sampler, signal_wrapper


class Minuit(Sampler):
    """bilby wrapper of iminuit (https://iminuit.readthedocs.io/)

    Provides a frequentist maximum-likelihood interface using the MINUIT
    minimisation algorithm. Runs MIGRAD to find the best-fit parameters,
    optionally HESSE for Hessian-based covariance, and optionally MINOS for
    profile-likelihood confidence intervals.

    "Samples" stored in the result are drawn from the multivariate Gaussian
    approximation centred on the best-fit point with the HESSE covariance.

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
        """Return the negative log-likelihood for minimisation."""
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

        # Build the initial parameter values from the prior means/modes.
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
            minos_ncall = self.kwargs["minos_ncall"]
            m.minos(ncall=minos_ncall)
            minos_errors = {
                key: (m.merrors[key].lower, m.merrors[key].upper)
                for key in self._search_parameter_keys
                if key in m.merrors
            }

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

        self.calc_likelihood_count()
        return self.result

    def _run_test(self):
        """Run a quick test with a small number of function calls."""
        self.kwargs["migrad_ncall"] = 10
        self.kwargs["run_hesse"] = False
        self.kwargs["run_minos"] = False
        self.kwargs["nsamples"] = 10
        return self.run_sampler()

    def write_current_state(self):
        """iminuit does not support checkpointing."""
        pass

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _initial_value(self, key):
        """Return a sensible starting value for parameter *key*.

        Uses the prior median (``rescale(0.5)``) when finite, otherwise falls
        back to 0.
        """
        prior = self.priors[key]
        try:
            val = float(prior.rescale(0.5))
        except (ValueError, TypeError, AttributeError):
            val = 0.0
        if not np.isfinite(val):
            val = 0.0
        return val

    def _prior_limits(self, key):
        """Return an (lower, upper) limit tuple for parameter *key*.

        Returns ``(None, None)`` when a bound is infinite.
        """
        prior = self.priors[key]
        lo = prior.minimum if np.isfinite(prior.minimum) else None
        hi = prior.maximum if np.isfinite(prior.maximum) else None
        return (lo, hi)
