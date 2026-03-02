import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import bilby
import bilby.core.sampler.minuit


class TestMinuitSampler(unittest.TestCase):
    def setUp(self):
        self.likelihood = MagicMock()
        self.likelihood.parameters = {}
        self.likelihood.log_likelihood_ratio = MagicMock(return_value=1.0)
        self.likelihood.log_likelihood = MagicMock(return_value=-1.0)
        self.likelihood.noise_log_likelihood = MagicMock(return_value=0.0)
        self.likelihood.meta_data = {}
        self.priors = bilby.core.prior.PriorDict(
            dict(
                a=bilby.core.prior.Uniform(0, 1, "a"),
                b=bilby.core.prior.Uniform(0, 1, "b"),
            )
        )
        self.sampler = bilby.core.sampler.minuit.Minuit(
            self.likelihood,
            self.priors,
            outdir="outdir",
            label="label",
            use_ratio=False,
            plot=False,
            skip_import_verification=True,
        )

    def tearDown(self):
        del self.likelihood
        del self.priors
        del self.sampler

    def test_default_kwargs(self):
        expected = dict(
            nsamples=1000,
            run_hesse=True,
            run_minos=False,
            compute_profiles=False,
            profile_size=100,
            start_from_result=None,
            migrad_ncall=None,
            minos_ncall=None,
            tol=None,
            strategy=1,
            print_level=0,
        )
        self.assertDictEqual(expected, self.sampler.kwargs)

    def test_sampler_name(self):
        self.assertEqual(self.sampler.sampler_name, "minuit")

    def test_initial_value_uniform(self):
        val = self.sampler._initial_value("a")
        self.assertAlmostEqual(val, 0.5)

    def test_initial_value_gaussian(self):
        self.priors["a"] = bilby.core.prior.Gaussian(3.0, 1.0, "a")
        self.sampler.priors = self.priors
        val = self.sampler._initial_value("a")
        self.assertAlmostEqual(val, 3.0)

    def test_prior_limits_uniform(self):
        lo, hi = self.sampler._prior_limits("a")
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_prior_limits_gaussian(self):
        self.priors["a"] = bilby.core.prior.Gaussian(0, 1, "a")
        self.sampler.priors = self.priors
        lo, hi = self.sampler._prior_limits("a")
        self.assertIsNone(lo)
        self.assertIsNone(hi)

    def test_neg_log_likelihood(self):
        theta = np.array([0.5, 0.5])
        result = self.sampler._neg_log_likelihood(theta)
        # Should be negative of log_likelihood
        self.assertIsInstance(result, float)

    def test_run_sampler_basic(self):
        """Test that run_sampler returns a result with expected attributes."""
        result = bilby.run_sampler(
            likelihood=bilby.core.likelihood.GaussianLikelihood(
                x=np.array([1.0, 2.0, 3.0]),
                y=np.array([0.5, 1.0, 1.5]),
                func=lambda x, m: m * x,
                sigma=0.1,
            ),
            priors=bilby.core.prior.PriorDict(
                {"m": bilby.core.prior.Uniform(0, 2, "m")}
            ),
            sampler="minuit",
            outdir="/tmp/test_minuit_basic",
            label="basic",
            save=False,
            nsamples=50,
            run_hesse=True,
            run_minos=False,
        )
        self.assertIn("best_fit", result.meta_data)
        self.assertIn("migrad_valid", result.meta_data)
        self.assertTrue(result.meta_data["migrad_valid"])
        self.assertAlmostEqual(result.meta_data["best_fit"]["m"], 0.5, places=2)
        self.assertEqual(result.posterior.shape[0], 50)
        self.assertIn("m", result.posterior.columns)

    def test_run_sampler_minos(self):
        """Test that run_sampler with MINOS stores minos_errors."""
        result = bilby.run_sampler(
            likelihood=bilby.core.likelihood.GaussianLikelihood(
                x=np.array([1.0, 2.0, 3.0]),
                y=np.array([0.5, 1.0, 1.5]),
                func=lambda x, m: m * x,
                sigma=0.1,
            ),
            priors=bilby.core.prior.PriorDict(
                {"m": bilby.core.prior.Uniform(0, 2, "m")}
            ),
            sampler="minuit",
            outdir="/tmp/test_minuit_minos",
            label="minos",
            save=False,
            nsamples=20,
            run_hesse=True,
            run_minos=True,
        )
        self.assertIn("minos_errors", result.meta_data)
        lo, hi = result.meta_data["minos_errors"]["m"]
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_verify_external_sampler_import_error(self):
        """Test that SamplerNotInstalledError is raised when iminuit is missing."""
        from bilby.core.sampler.base_sampler import SamplerNotInstalledError

        with patch.dict("sys.modules", {"iminuit": None}):
            with self.assertRaises((SamplerNotInstalledError, ImportError)):
                self.sampler._verify_external_sampler()

    def test_write_current_state_is_noop(self):
        """write_current_state should not raise."""
        self.sampler.write_current_state()  # must not raise

    def test_run_test(self):
        """_run_test should complete quickly with reduced settings."""
        likelihood = bilby.core.likelihood.GaussianLikelihood(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0, 2.0]),
            func=lambda x, m: m * x,
            sigma=0.1,
        )
        priors = bilby.core.prior.PriorDict(
            {"m": bilby.core.prior.Uniform(0, 3, "m")}
        )
        sampler = bilby.core.sampler.minuit.Minuit(
            likelihood,
            priors,
            outdir="/tmp/test_minuit_runtest",
            label="runtest",
            use_ratio=False,
            skip_import_verification=False,
        )
        result = sampler._run_test()
        self.assertIsNotNone(result)

    def test_run_sampler_profiles(self):
        """Test that compute_profiles stores profile likelihood in meta_data."""
        result = bilby.run_sampler(
            likelihood=bilby.core.likelihood.GaussianLikelihood(
                x=np.array([1.0, 2.0, 3.0]),
                y=np.array([0.5, 1.0, 1.5]),
                func=lambda x, m: m * x,
                sigma=0.1,
            ),
            priors=bilby.core.prior.PriorDict(
                {"m": bilby.core.prior.Uniform(0, 2, "m")}
            ),
            sampler="minuit",
            outdir="/tmp/test_minuit_profiles",
            label="profiles",
            save=False,
            nsamples=20,
            run_hesse=True,
            compute_profiles=True,
            profile_size=10,
        )
        self.assertIn("profiles", result.meta_data)
        self.assertIn("m", result.meta_data["profiles"])
        profile = result.meta_data["profiles"]["m"]
        self.assertIn("values", profile)
        self.assertIn("log_likelihood", profile)
        self.assertEqual(len(profile["values"]), 10)
        # Maximum log_likelihood in profile should be near 0 (at the MLE);
        # allow a small tolerance since the scan may not hit exactly the MLE.
        self.assertGreater(max(profile["log_likelihood"]), -0.1)
        # Profile must decrease away from the peak: values at the edges
        # should be lower than the maximum.
        logl = np.array(profile["log_likelihood"])
        max_idx = np.argmax(logl)
        # At least one point to the left and one to the right should be lower.
        if max_idx > 0:
            self.assertLess(logl[0], logl[max_idx])
        if max_idx < len(logl) - 1:
            self.assertLess(logl[-1], logl[max_idx])
        # Profile values should span a meaningful range around the MLE.
        mle = result.meta_data["best_fit"]["m"]
        self.assertLess(min(profile["values"]), mle)
        self.assertGreater(max(profile["values"]), mle)

    def test_warm_start_from_result(self):
        """Test that start_from_result uses the MAP from the previous result."""
        import os
        import tempfile

        # First run to produce a reference result.
        likelihood = bilby.core.likelihood.GaussianLikelihood(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([0.5, 1.0, 1.5]),
            func=lambda x, m: m * x,
            sigma=0.1,
        )
        priors = bilby.core.prior.PriorDict(
            {"m": bilby.core.prior.Uniform(0, 2, "m")}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            prior_result = bilby.run_sampler(
                likelihood=likelihood,
                priors=priors,
                sampler="minuit",
                outdir=tmpdir,
                label="warm_prior",
                save=True,
                nsamples=20,
            )
            prior_result_file = os.path.join(tmpdir, "warm_prior_result.json")

            # Warm-start run using the saved result file.
            result = bilby.run_sampler(
                likelihood=likelihood,
                priors=priors,
                sampler="minuit",
                outdir=tmpdir,
                label="warm_result",
                save=False,
                nsamples=20,
                start_from_result=prior_result_file,
            )
        self.assertIn("best_fit", result.meta_data)
        self.assertAlmostEqual(result.meta_data["best_fit"]["m"], 0.5, places=2)

    def test_warm_start_from_result_object(self):
        """Test warm start from a Result object (not a file path)."""
        likelihood = bilby.core.likelihood.GaussianLikelihood(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([0.5, 1.0, 1.5]),
            func=lambda x, m: m * x,
            sigma=0.1,
        )
        priors = bilby.core.prior.PriorDict(
            {"m": bilby.core.prior.Uniform(0, 2, "m")}
        )
        prior_result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="minuit",
            outdir="/tmp/test_warm_obj",
            label="warm_obj_prior",
            save=False,
            nsamples=20,
        )
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="minuit",
            outdir="/tmp/test_warm_obj",
            label="warm_obj_result",
            save=False,
            nsamples=20,
            start_from_result=prior_result,
        )
        self.assertIn("best_fit", result.meta_data)
        self.assertAlmostEqual(result.meta_data["best_fit"]["m"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
