"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1,3],[2,13],[3,5]],[2,7]),         # normal input
        ([[1,3],[1,3],[1,3]],[1,3]),            # single value for all cases
        ([[-12,-3],[-2,-3],[-1,-3]],[-5,-3]),  # only negative values
        ([[1,30],[9,-31],[-1,1]],[3,0]),    # values above and below zero
        ([[0,0],[0,0],[0,0]],[0,0])             # only zero
    ]
)


def test_daily_mean_parameterized(test, expected):
    """Additional tests of daily_mean() using parameterization"""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


def test_daily_max_same():
    """Test that max function works when only passed a single repeated integer"""

    test_input = np.array([[1,3],
                          [1,3],
                          [1,3]])
    test_result = np.array([1,3])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_varied():
    """Test that max function works when passed a variety of integers"""

    test_input = np.array([[1, 3],
                           [2, 11],
                           [3, 5]])
    test_result = np.array([3, 11])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_negative():
    """Test that max function works when only passed negative integers"""

    test_input = np.array([[-12, -3],
                           [-21, -3],
                           [-1, -3]])
    test_result = np.array([-1, -3])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_max_mixed():
    """Test that max function works when passed positive and negative integers"""

    test_input = np.array([[1, 30],
                           [10, -999],
                           [-1, 3]])
    test_result = np.array([10, 30])

    npt.assert_array_equal(daily_max(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1,3],[2,11],[3,5]],[3,11]),          # normal input
        ([[1,3],[1,3],[1,3]],[1,3]),            # single value for all cases
        ([[-12,-3],[-21,-3],[-1,-3]],[-1,-3]),  # only negative values
        ([[1,30],[10,-999],[-1,3]],[10,30]),    # values above and below zero
        ([[0,0],[0,0],[0,0]],[0,0])             # only zero
    ]
)


def test_daily_max_parameterized(test, expected):
    """Additional tests of daily_max() using parameterization"""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


def test_daily_min_same():
    """Test that min function works when only passed a single repeated integer"""

    test_input = np.array([[1,3],
                          [1,3],
                          [1,3]])
    test_result = np.array([1,3])

    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_varied():
    """Test that min function works when passed a variety of integers"""

    test_input = np.array([[1, 3],
                           [2, 11],
                           [3, 5]])
    test_result = np.array([1, 3])

    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_negative():
    """Test that min function works when only passed negative integers"""

    test_input = np.array([[-12, -3],
                           [-21, -3],
                           [-1, -3]])
    test_result = np.array([-21, -3])

    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_mixed():
    """Test that min function works when passed positive and negative integers"""

    test_input = np.array([[1, 30],
                           [10, -999],
                           [-1, 3]])
    test_result = np.array([-1, -999])

    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1,3],[2,11],[3,5]],[1,3]),           # normal input
        ([[1,3],[1,3],[1,3]],[1,3]),            # single value for all cases
        ([[-12,-3],[-21,-3],[-1,-3]],[-21,-3]), # only negative values
        ([[1,30],[10,-999],[-1,3]],[-1,-999]),  # values above and below zero
        ([[0,0],[0,0],[0,0]],[0,0])             # only zero
    ]
)


def test_daily_min_parameterized(test, expected):
    """Additional tests of daily_min() using parameterization"""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))