"""Small local assertions used by the original optimization tutorial tests."""

import numpy as np
from copy import deepcopy


def _compare(actual, expected, check):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            _compare(actual[key], expected[key], check)
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _compare(left, right, check)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        if check != "datatype_check":
            assert actual.shape == expected.shape
        if check == "equation_output_check":
            assert np.allclose(actual, expected)
    else:
        assert type(actual) is type(expected)
        if check == "equation_output_check":
            assert np.isclose(actual, expected)


def multiple_test(test_cases, target):
    for case in test_cases:
        actual = target(*deepcopy(case["input"]))
        try:
            _compare(actual, case["expected"], case["name"])
        except AssertionError as exc:
            raise AssertionError(f'{case["name"]}: {case["error"]}') from exc
    print("All tests passed")


def single_test(test_case, target):
    return multiple_test([test_case], target)
