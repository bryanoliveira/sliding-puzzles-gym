import unittest
from sliding_puzzles.env import SlidingEnv


class TestSlidingEnv(unittest.TestCase):
    def setUp(self):
        self.env_w = 4
        self.env_h = 4
        self.env_blank_value = -1
        self.env = SlidingEnv(
            render_mode="state",
            blank_value=self.env_blank_value,
            w=self.env_w,
            h=self.env_h,
        )

    def test_initial_state_no_shuffle(self):
        env = SlidingEnv(
            render_mode="state",
            blank_value=self.env_blank_value,
            w=self.env_w,
            h=self.env_h,
            shuffle_steps=0,
        )

        # Test that the initial state is as expected
        initial_state, initial_info = env.reset()
        expected_state = [
            [-1, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
        self.assertEqual(initial_state.tolist(), expected_state)
        self.assertFalse(initial_info["is_success"])

        env.close()

    def test_initial_state_shuffle(self):
        env = SlidingEnv(
            render_mode="state",
            blank_value=self.env_blank_value,
            w=self.env_w,
            h=self.env_h,
            shuffle_steps=10,
        )

        initial_state, initial_info = env.reset()

        # the initial state will be shuffled, so we have to check if every number is in the state
        self.assertIn(self.env_blank_value, initial_state)
        for i in range(1, 16):
            self.assertIn(i, initial_state)
        # check if key is_success is false in info
        self.assertFalse(initial_info["is_success"])

        env.close()


    def test_step(self):
        env = SlidingEnv(
            render_mode="state",
            blank_value=self.env_blank_value,
            w=self.env_w,
            h=self.env_h,
            shuffle_steps=0,
            sparse_rewards=True,
            invalid_move_reward=-1,
            move_reward=-0.5,
            render_shuffling=True,
        )

        # Test that the state changes as expected after a step
        env.reset()
        initial_state = env.state.copy()
        state, reward, _, _, _ = env.step(0)  # Move the bottom tile up
        self.assertNotEqual(state.tolist(), initial_state.tolist())
        self.assertEqual(reward, -0.5)  # Assuming the reward for a move is -1

        env.close()

    def test_reward(self):
        env = SlidingEnv(
            render_mode="state",
            blank_value=self.env_blank_value,
            w=self.env_w,
            h=self.env_h,
            shuffle_steps=0,
            sparse_rewards=True,
            invalid_move_reward=-1,
            move_reward=-0.5,
            render_shuffling=True,
        )

        # Test that the reward is as expected after a step
        env.reset()
        _, reward, _, _, _ = env.step(0)  # Move the bottom tile up
        self.assertEqual(reward, -0.5)
        _, reward, _, _, _ = env.step(1)  # Move the left tile right (invalid)
        self.assertEqual(reward, -1)

        env.close()


    def tearDown(self):
        self.env.close()


if __name__ == "__main__":
    unittest.main()
