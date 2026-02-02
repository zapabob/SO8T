import json
from pathlib import Path

from scripts.evaluation import deepeval_ethics_test as de


def test_sanitize_name_replaces_forbidden_chars():
    assert de.sanitize_name("a/b:c d") == "a_b_c_d"


def test_load_cases_from_file(tmp_path):
    cases_path = tmp_path / "cases.json"
    payload = [{"name": "case1", "prompt": "hello"}]
    cases_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = de.load_cases(cases_path)
    assert loaded[0]["name"] == "case1"
    assert loaded[0]["prompt"] == "hello"


def test_summarize_counts_pass_rates():
    results = [
        {
            "name": "case1",
            "prompt": "p",
            "actual_output": "o",
            "metrics": [
                {"metric": "HallucinationMetric", "passed": True},
                {"metric": "BiasMetric", "passed": False},
            ],
        },
        {
            "name": "case2",
            "prompt": "p",
            "actual_output": "o",
            "metrics": [
                {"metric": "HallucinationMetric", "passed": False},
                {"metric": "BiasMetric", "passed": True},
            ],
        },
    ]
    summary = de.summarize(results)
    assert summary["total_cases"] == 2
    assert summary["passing_rates"]["HallucinationMetric"] == 0.5
    assert summary["passing_rates"]["BiasMetric"] == 0.5

