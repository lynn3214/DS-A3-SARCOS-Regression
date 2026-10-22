"""Synthetic tests for out-of-range and coincident error-slice boundaries."""
import unittest

import numpy as np

from scripts.analysis_helpers import make_test_error_slices


class ErrorSliceTests(unittest.TestCase):
    def test_open_ended_intervals_and_outside_counts(self):
        train_x = np.zeros((4, 21), dtype=float)
        train_x[:, 7] = [1, 2, 3, 4]
        train_x[:, 14] = [1, 2, 3, 4]
        train_y = np.array([1, 2, 3, 4], dtype=float)

        test_x = np.zeros((2, 21), dtype=float)
        test_x[:, 7] = [0.5, 4.5]
        test_x[:, 14] = [0.5, 4.5]
        test_y = np.array([0.5, 4.5])
        prediction = np.zeros(2)

        table, serialized = make_test_error_slices(
            train_x, train_y, test_x, test_y, prediction
        )

        for key in ["velocity_norm", "acceleration_norm", "absolute_torque"]:
            self.assertEqual(serialized[key]["outside_training_range_count"], 2)
            self.assertEqual(serialized[key]["effective_slice_count"], 4)

        velocity_rows = table[table["Slice variable"] == "Velocity norm"]
        interval_column = "Slice interval (train-derived cutoffs)"
        self.assertEqual(velocity_rows.iloc[0][interval_column], "(-inf, 1.75]")
        self.assertEqual(velocity_rows.iloc[-1][interval_column], "(3.25, +inf)")
        self.assertEqual(velocity_rows.iloc[0]["n"], 1)
        self.assertEqual(velocity_rows.iloc[-1]["n"], 1)

    def test_coincident_quantile_boundaries_are_collapsed(self):
        train_x = np.zeros((4, 21), dtype=float)
        train_x[:, 7] = [1, 1, 1, 2]
        train_x[:, 14] = [1, 1, 1, 2]
        train_y = np.array([1, 1, 1, 2], dtype=float)
        test_x = np.zeros((2, 21), dtype=float)
        test_x[:, 7] = [1.1, 1.8]
        test_x[:, 14] = [1.1, 1.8]
        test_y = np.array([1.1, 1.8])

        table, serialized = make_test_error_slices(
            train_x, train_y, test_x, test_y, np.zeros(2)
        )

        self.assertEqual(serialized["velocity_norm"]["training_quartile_edges"], [1.0, 1.25, 2.0])
        self.assertEqual(serialized["velocity_norm"]["effective_slice_count"], 2)
        velocity_rows = table[table["Slice variable"] == "Velocity norm"]
        interval_column = "Slice interval (train-derived cutoffs)"
        self.assertEqual(
            velocity_rows[interval_column].tolist(),
            ["(-inf, 1.25]", "(1.25, +inf)"],
        )
        self.assertEqual(int(velocity_rows["n"].sum()), len(test_y))

    def test_constant_training_measure_uses_one_all_values_slice(self):
        train_x = np.zeros((4, 21), dtype=float)
        train_x[:, 7] = 1
        train_x[:, 14] = 1
        train_y = np.ones(4)
        test_x = np.zeros((2, 21), dtype=float)
        test_x[:, 7] = [0, 3]
        test_x[:, 14] = [0, 3]
        test_y = np.array([0.5, 2.0])

        table, serialized = make_test_error_slices(
            train_x, train_y, test_x, test_y, np.zeros(2)
        )

        self.assertEqual(serialized["velocity_norm"]["effective_slice_count"], 1)
        self.assertEqual(
            serialized["velocity_norm"]["test_bins"][0]["interval_label"],
            "(-inf, +inf)",
        )
        self.assertEqual(serialized["velocity_norm"]["outside_training_range_count"], 2)
        self.assertEqual(
            table.loc[table["Slice variable"] == "Velocity norm", "n"].sum(),
            len(test_y),
        )


if __name__ == "__main__":
    unittest.main()
