import unittest

from vla_bridge import JOINT_NAMES, ReadOnlyVLABridge, format_positions


class ReadOnlyVLABridgeTest(unittest.TestCase):
    def test_mock_returns_six_named_positions(self):
        positions = ReadOnlyVLABridge().predict()

        self.assertEqual(tuple(positions), JOINT_NAMES)
        self.assertEqual(list(positions.values()), [0.0] * 6)
        self.assertIn("gripper.pos=0.000", format_positions(positions))

    def test_invalid_prediction_is_rejected(self):
        bridge = ReadOnlyVLABridge()
        bridge._predict_action = lambda: [0.0]

        with self.assertRaisesRegex(ValueError, "six finite values"):
            bridge.predict()


if __name__ == "__main__":
    unittest.main()
