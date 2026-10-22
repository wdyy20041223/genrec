from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np

from .analyze_sid_similarity import (
    build_pair_spaces, build_train_item_users, collaborative_cosine,
    cointeraction_rate, lcp_length, normalize_embeddings, numeric_stats,
    sample_pairs, semantic_cosine,
)


def test_exact_lcp_ignores_disambiguation_suffix():
    left = np.array([[12, 35, 80, 0]] * 4)
    right = np.array([[8, 35, 80, 0], [12, 41, 80, 0], [12, 35, 91, 0], [12, 35, 80, 1]])
    assert lcp_length(left, right).tolist() == [0, 1, 2, 3]


def test_pair_rank_spaces_equal_independent_brute_force_partition():
    codes = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 0],
                      [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 1, 0, 0]])
    item_ids = np.array([18, 3, 4, 99, 12, 7, 50, 1])
    spaces = build_pair_spaces(codes, item_ids)
    truth = [set() for _ in range(4)]
    for i, j in combinations(range(len(codes)), 2):
        prefix = 0
        for a, b in zip(codes[i, :3], codes[j, :3]):
            if a != b:
                break
            prefix += 1
        truth[prefix].add(tuple(sorted((int(item_ids[i]), int(item_ids[j])))))
    for k, space in enumerate(spaces):
        pairs = space.pairs_at(np.arange(space.total))
        actual = set(map(tuple, pairs.tolist()))
        assert actual == truth[k]
        assert len(actual) == space.total
    assert sum(space.total for space in spaces) == 28
    assert spaces[3].total == 3


def test_sample_without_replacement_exact_groups_and_reproducibility():
    codes = np.random.default_rng(7).integers(0, 3, size=(30, 3))
    for k, space in enumerate(build_pair_spaces(codes)):
        a = sample_pairs(space, 12, np.random.default_rng(42))
        b = sample_pairs(space, 12, np.random.default_rng(42))
        assert np.array_equal(a, b)
        assert len(a) == min(12, space.total)
        assert len(set(map(tuple, a.tolist()))) == len(a)
        assert (a[:, 0] < a[:, 1]).all()
        assert (lcp_length(codes[a[:, 0]], codes[a[:, 1]]) == k).all()


def test_zero_pair_groups_and_small_groups_are_not_fabricated():
    codes = np.array([[0, 0, 0], [1, 0, 0]])
    spaces = build_pair_spaces(codes)
    assert [s.total for s in spaces] == [1, 0, 0, 0]
    assert sample_pairs(spaces[0], 1000000, np.random.default_rng(42)).tolist() == [[0, 1]]
    assert sample_pairs(spaces[3], 1000000, np.random.default_rng(42)).shape == (0, 2)
    assert all(np.isnan(v) for v in numeric_stats([]))
    assert np.isnan(cointeraction_rate(np.array([]), np.array([], dtype=bool)))


def test_training_deduplication_holdout_exclusion_and_missing_interactions():
    sequences = [np.array([0, 0, 1, 2, 3]), np.array([0, 2, 4, 1, 3])]
    matrix, degrees = build_train_item_users(sequences, 5)
    assert degrees.tolist() == [2, 1, 1, 0, 1]
    assert matrix.nnz == 5
    assert matrix[1, 1] == 0  # Validation occurrence is excluded.
    assert matrix[3].nnz == 0  # Test-only item remains undefined for cosine.
    left, right = np.array([0, 1, 0]), np.array([1, 2, 3])
    common = np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()
    cosine, valid = collaborative_cosine(common, degrees[left], degrees[right])
    assert common.tolist() == [1, 0, 0]
    assert valid.tolist() == [True, True, False]
    assert np.isclose(cosine[0], 1 / np.sqrt(2))
    assert cosine[1] == 0
    assert np.isnan(cosine[2])
    assert cointeraction_rate(common, valid) == 0.5


def test_common_user_count_does_not_overflow_uint8():
    matrix, degrees = build_train_item_users([np.array([0, 1, 2, 3, 3])] * 512, 4)
    shared = np.asarray(matrix[[0]].multiply(matrix[[1]]).sum(axis=1)).ravel()
    assert shared.tolist() == [512]
    cosine, valid = collaborative_cosine(shared, degrees[[0]], degrees[[1]])
    assert valid.tolist() == [True]
    assert cosine.tolist() == [1.0]


def test_semantic_cosine_and_invalid_embedding_filter():
    values = np.array([[1., 0.], [0., 2.], [1., 1.], [0., 0.], [np.nan, 1.]])
    normalized, valid = normalize_embeddings(values)
    assert valid.tolist() == [True, True, True, False, False]
    result = semantic_cosine(normalized, np.array([0, 0, 1]), np.array([1, 2, 2]))
    assert np.allclose(result, [0., 1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert np.isnan(normalized[3:]).all()


def test_ambiguous_item_mapping_is_rejected():
    with unittest.TestCase().assertRaisesRegex(ValueError, "unique"):
        build_pair_spaces(np.array([[0, 0, 0], [1, 0, 0]]), np.array([3, 3]))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(function)
        for name, function in sorted(globals().items()) if name.startswith("test_")
    )


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(load_tests(None, None, None))
    sys.exit(not result.wasSuccessful())
