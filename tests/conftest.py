"""Public export: skip the tests that check internal audit documents.

These tests compare generated German audit reports, the protocol document and
blueprint reference code with the committed data. Those files are kept in the
private research repository and are not part of this export. Every other test
runs unchanged.
"""

from __future__ import annotations

import pytest

PRIVATE_DOCUMENTS = {
    "tests/e01/test_t04.py::test_table_in_document_is_generated_from_raw_outputs",
    "tests/e02/test_t01.py::test_schema_matches_blueprint_reference_keyed_uniform",
    "tests/e06/test_t03.py::test_report_is_generated_from_current_results",
    "tests/e06/test_t03.py::test_h3_is_not_claimed_and_disclaimer_is_present",
    "tests/e06/test_t03.py::test_negative_cases_and_unconfirmed_expectation_are_documented",
    "tests/e07/test_t06.py::test_configs_and_lists_are_hashed_and_the_document_names_the_hashes",
    "tests/e07/test_t06.py::test_novelty_and_claim_rules_are_frozen",
    "tests/e07/test_t06.py::test_committed_tuning_result_reproduces_the_selection",
    "tests/e08/test_t01.py::test_committed_pilot_reproduces_the_plan",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.nodeid in PRIVATE_DOCUMENTS:
            item.add_marker(
                pytest.mark.skip(reason="needs internal audit documents (private research repo)")
            )
