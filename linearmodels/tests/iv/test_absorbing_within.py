"""
AbsorbingLS.fit with method="within"

within (https://github.com/py-econometrics/within) is an optional dependency
that absorbs the variables using a preconditioned LSMR solver written in Rust.
It absorbs all of the variables that AbsorbingLS can absorb: categorical
effects, continuous variables and interactions, in weighted and unweighted
models. The default method, "auto", uses it when it is installed and
absorb_options is empty.
"""

import types

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import absorbing
from linearmodels.iv.absorbing import (
    _VARIABLE_CACHE,
    AbsorbingLS,
    Interaction,
    clear_cache,
)
from linearmodels.panel.utility import AbsorbingEffectWarning
from linearmodels.shared.utility import AttrDict


@pytest.fixture
def requires_within():
    return pytest.importorskip("within")


NOBS = 2000
CASES = ["cats", "cats_cont", "interactions", "weighted", "weighted_interactions"]
TIGHT_LSMR = {"atol": 1e-12, "btol": 1e-12}
TIGHT_WITHIN = {"tol": 1e-11}


@pytest.fixture(autouse=True)
def empty_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": pd.Categorical(rng.integers(0, 100, NOBS)),
            "b": pd.Categorical(rng.integers(0, 20, NOBS)),
            "c": rng.standard_normal(NOBS),
            "d": rng.standard_normal(NOBS),
        }
    )
    x = pd.DataFrame(
        {
            "const": 1.0,
            "x1": rng.standard_normal(NOBS),
            "x2": rng.standard_normal(NOBS),
        }
    )
    y = x.x1 + frame.c + rng.standard_normal(NOBS)
    weights = rng.uniform(0.5, 2.0, NOBS)
    return AttrDict(frame=frame, x=x, y=y, weights=weights)


def make_model(data, case):
    frame = data.frame
    if case == "cats":
        kwargs = {"absorb": frame[["a", "b"]]}
    elif case == "cats_cont":
        kwargs = {"absorb": frame[["a", "b", "c"]]}
    elif case == "interactions":
        kwargs = {
            "absorb": frame[["a"]],
            "interactions": [
                Interaction(frame[["b"]], frame[["c", "d"]]),
                Interaction(frame[["a", "b"]]),
                Interaction(cont=frame[["d"]]),
            ],
        }
    elif case == "weighted":
        kwargs = {"absorb": frame[["a", "b"]], "weights": data.weights}
    else:
        kwargs = {
            "absorb": frame[["a", "c"]],
            "interactions": Interaction(frame[["b"]], frame[["d"]]),
            "weights": data.weights,
        }
    return AbsorbingLS(data.y, data.x, **kwargs)


def assert_results_close(res, ref, rtol=1e-7):
    assert_allclose(res.params, ref.params, rtol=rtol, atol=1e-10)
    assert_allclose(res.cov, ref.cov, rtol=rtol, atol=1e-12)
    assert_allclose(res.resids, ref.resids, rtol=rtol, atol=1e-8)
    assert res.df_model == ref.df_model
    assert_allclose(res.rsquared, ref.rsquared, rtol=rtol)
    assert_allclose(res.absorbed_rsquared, ref.absorbed_rsquared, rtol=rtol)


@pytest.mark.usefixtures("requires_within")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
def test_within_matches_lsmr(data, case, cov_type):
    ref = make_model(data, case).fit(
        method="lsmr", absorb_options=TIGHT_LSMR, cov_type=cov_type
    )
    res = make_model(data, case).fit(
        method="within", absorb_options=TIGHT_WITHIN, cov_type=cov_type
    )
    assert_results_close(res, ref)


@pytest.mark.usefixtures("requires_within")
def test_within_matches_hdfe(data):
    ref = make_model(data, "cats").fit(
        method="hdfe", absorb_options={"options": {"tol": 1e-14}}
    )
    res = make_model(data, "cats").fit(method="within", absorb_options=TIGHT_WITHIN)
    assert_results_close(res, ref)


@pytest.mark.usefixtures("requires_within")
@pytest.mark.parametrize("case", CASES)
def test_within_absorbed_variables(data, case):
    mod = make_model(data, case)
    mod.fit(method="within", absorb_options=TIGHT_WITHIN)
    ref = make_model(data, case)
    ref.fit(method="lsmr", absorb_options=TIGHT_LSMR)
    assert_allclose(mod.absorbed_dependent, ref.absorbed_dependent, atol=1e-8)
    assert_allclose(mod.absorbed_exog, ref.absorbed_exog, atol=1e-8)


@pytest.mark.usefixtures("requires_within")
def test_method_is_case_insensitive(data):
    res = make_model(data, "cats").fit(method="WITHIN")
    ref = make_model(data, "cats").fit(method="within")
    assert_allclose(res.params, ref.params)


@pytest.fixture
def algorithms(monkeypatch):
    """Record which algorithm absorbs the variables"""
    used = []
    original_within = absorbing.within_annihilate
    original_lsmr = absorbing.lsmr_annihilate

    def spy_within(*args, **kwargs):
        used.append("within")
        return original_within(*args, **kwargs)

    def spy_lsmr(*args, **kwargs):
        used.append("lsmr")
        return original_lsmr(*args, **kwargs)

    monkeypatch.setattr(absorbing, "within_annihilate", spy_within)
    monkeypatch.setattr(absorbing, "lsmr_annihilate", spy_lsmr)
    return used


@pytest.mark.usefixtures("requires_within")
@pytest.mark.parametrize("case", CASES)
def test_auto_uses_within(data, case, algorithms):
    make_model(data, case).fit()
    assert algorithms == ["within"]


@pytest.mark.usefixtures("requires_within")
def test_auto_with_options_does_not_use_within(data, algorithms):
    # Options of LSMR or HDFE select the algorithm as if within were not installed
    make_model(data, "weighted").fit(absorb_options={"atol": 1e-10})
    assert algorithms == ["lsmr"]


def test_auto_without_within(data, algorithms, monkeypatch):
    monkeypatch.setattr(absorbing, "HAVE_WITHIN", False)
    make_model(data, "weighted").fit()
    assert algorithms == ["lsmr"]


def test_within_not_installed(data, monkeypatch):
    monkeypatch.setattr(absorbing, "HAVE_WITHIN", False)
    with pytest.raises(ImportError, match="pip install within-py"):
        make_model(data, "cats").fit(method="within")


@pytest.mark.usefixtures("requires_within")
@pytest.mark.parametrize("options", [{"atol": 1e-8}, {"tol": 1e-8, "foo": 1}])
def test_unknown_within_options(data, options):
    mod = make_model(data, "cats")
    with pytest.raises(ValueError, match="that the within solver does not have"):
        mod.fit(method="within", absorb_options=options)


@pytest.mark.usefixtures("requires_within")
def test_within_options_reach_the_solver(data):
    # One iteration does not converge, which warns and gives other residuals
    mod = make_model(data, "interactions")
    with pytest.warns(RuntimeWarning, match="within solver did not converge"):
        limited = mod.fit(method="within", absorb_options={"maxiter": 1})
    default = make_model(data, "interactions").fit(method="within")
    assert not np.allclose(limited.params, default.params)


@pytest.mark.usefixtures("requires_within")
def test_preconditioner_option(data, requires_within):
    within = requires_within

    ref = make_model(data, "interactions").fit(
        method="within", absorb_options=TIGHT_WITHIN
    )
    for config in (
        within.PreconditionerConfig.Diagonal(),
        within.PreconditionerConfig.Additive(),
    ):
        res = make_model(data, "interactions").fit(
            method="within",
            absorb_options={**TIGHT_WITHIN, "preconditioner": config},
        )
        assert_results_close(res, ref, rtol=1e-6)
    # Preconditioners cannot be compared, so their residuals are not cached
    assert all(
        key[0] != "within" or "preconditioner" not in str(key)
        for key in _VARIABLE_CACHE
    )


@pytest.fixture
def count_solves(monkeypatch, requires_within):
    """Record the columns that the within solver absorbs"""
    within = requires_within

    columns = []

    class CountingSolver:
        def __init__(self, *args, **kwargs):
            self._solver = within.Solver(*args, **kwargs)

        def solve_batch(self, y, options=None):
            columns.append(y.shape[1])
            return self._solver.solve_batch(y, options)

    fake = types.SimpleNamespace(
        Solver=CountingSolver, LsmrOptions=within.LsmrOptions, Effect=within.Effect
    )
    monkeypatch.setattr(absorbing, "within", fake)
    return columns


@pytest.mark.usefixtures("requires_within")
def test_within_cache(data, count_solves):
    first = make_model(data, "weighted").fit(method="within")
    assert count_solves == [4]
    # The same variables and effects, so all residuals come from the cache
    second = make_model(data, "weighted").fit(method="within")
    assert count_solves == [4]
    assert_allclose(first.params, second.params)
    # Other options are not the same residuals
    make_model(data, "weighted").fit(method="within", absorb_options={"tol": 1e-6})
    assert count_solves == [4, 4]
    # Only the new column of exog is absorbed
    x = data.x.copy()
    x["x3"] = np.arange(NOBS, dtype=float) % 7
    AbsorbingLS(data.y, x, absorb=data.frame[["a", "b"]], weights=data.weights).fit(
        method="within"
    )
    assert count_solves == [4, 4, 1]


@pytest.mark.usefixtures("requires_within")
def test_within_without_cache(data, count_solves):
    make_model(data, "cats").fit(method="within", use_cache=False)
    make_model(data, "cats").fit(method="within", use_cache=False)
    assert count_solves == [4, 4]
    assert len(_VARIABLE_CACHE) == 0


@pytest.mark.usefixtures("requires_within")
def test_refit_with_other_method(data, algorithms):
    mod = make_model(data, "weighted")
    res_within = mod.fit(method="within")
    res_lsmr = mod.fit(method="lsmr")
    assert algorithms == ["within", "lsmr"]
    assert_allclose(res_within.params, res_lsmr.params, rtol=1e-6)
    mod.fit(method="within", cov_type="unadjusted")
    assert algorithms == ["within", "lsmr", "within"]


@pytest.mark.usefixtures("requires_within")
def test_within_drop_absorbed(data):
    x = data.x.copy()
    x["absorbed"] = data.frame.c
    mod = AbsorbingLS(data.y, x, absorb=data.frame[["a", "c"]], drop_absorbed=True)
    with pytest.warns(AbsorbingEffectWarning, match="absorbed"):
        res = mod.fit(method="within")
    assert "absorbed" not in res.params.index
