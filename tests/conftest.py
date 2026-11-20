"""Shared fixtures."""

import pytest
from fakes import Oracle

from regenbench.adapters import HashAdapter
from regenbench.evaluate import evaluate_suite
from regenbench.generate import write_generated_suite
from regenbench.items import read_suite
from regenbench.summarise import summarise


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    """A small study with fake models whose behaviour is known: three suites, four models, summarised."""
    tmp = tmp_path_factory.mktemp("study")
    suites = []
    for seed in (0, 1, 2):
        write_generated_suite(tmp / "data", "vision", seed=seed, n_per_task=36, n_templates=3 if seed == 0 else 1)
        suites.append(tmp / "data" / f"vision-seed{seed}")
    items = [it for d in suites for it in read_suite(d)[1]]
    models = [
        Oracle(items, name="strong", skill=0.9),
        Oracle(items, name="middling", skill=0.65),
        Oracle(items, name="fragile", skill=0.8, weak_template=2),
        HashAdapter(name="guesser"),
    ]
    for model in models:
        for d in suites:
            evaluate_suite(model, d, tmp / "results", "t", cache_root=tmp / "cache")
    return tmp, summarise(tmp / "results", "t")
