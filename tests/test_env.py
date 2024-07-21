import numpy as np
import pickle
import unittest
import sliding_puzzles


class TestSlidingEnv(unittest.TestCase):
    def setUp(self):
        self.env_w = 4

    def test_default_values(self):
        env = sliding_puzzles.make(
            w=self.env_w,
        )
        self.assertEqual(env.grid_size_h, self.env_w)
        self.assertEqual(env.grid_size_w, self.env_w)
        self.assertEqual(env.render_mode, "state")
        self.assertEqual(env.sparse_rewards, False)
        self.assertEqual(env.win_reward, 10)
        self.assertEqual(env.invalid_move_reward, -1)
        self.assertEqual(env.move_reward, -0.1)
        self.assertEqual(env.reward_mode, "distances")
        self.assertEqual(env.circular_actions, False)
        self.assertEqual(env.blank_value, -1)
        self.assertEqual(env.shuffle_mode, "fast")

    def test_initial_state_no_shuffle(self):
        env = sliding_puzzles.make(
            w=self.env_w,
            shuffle_mode="serial",
            shuffle_steps=0,
        )

        # Test that the initial state is as expected
        initial_state, initial_info = env.reset()
        expected_state = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, -1],
        ]
        self.assertEqual(initial_state.tolist(), expected_state)
        self.assertFalse(initial_info["is_success"])

        env.close()

    def test_initial_state_shuffle(self):
        blank_value = -1
        env = sliding_puzzles.make(
            render_mode="state",
            blank_value=blank_value,
            w=self.env_w,
            shuffle_mode="fast",
        )

        initial_state, initial_info = env.reset()

        # the initial state will be shuffled, so we have to check if every number is in the state
        self.assertIn(blank_value, initial_state)
        for i in range(1, 16):
            self.assertIn(i, initial_state)
        # check if key is_success is false in info
        self.assertFalse(initial_info["is_success"])

        # Check if the state is different from the solved state
        solved_state = np.arange(1, 17).reshape(4, 4)
        solved_state[-1, -1] = blank_value
        self.assertFalse(np.array_equal(initial_state, solved_state))

        env.close()

    def test_reward(self):
        env = sliding_puzzles.make(
            w=self.env_w,
            shuffle_mode="serial",
            shuffle_steps=0,
            sparse_rewards=True,
            invalid_move_reward=-1,
            move_reward=-0.1,
        )

        # Test that the reward is as expected after a step
        env.reset()
        _, reward, _, _, _ = env.step(1)  # Move the top tile down
        self.assertEqual(reward, -0.1)
        _, reward, _, _, _ = env.step(2)  # Move the right tile left (invalid)
        self.assertEqual(reward, -1)
        _, reward, _, _, _ = env.step(3)  # Move the left tile right
        self.assertEqual(reward, -0.1)
        _, reward, _, _, _ = env.step(0)  # Move the bottom tile up
        self.assertEqual(reward, -0.1)

        env.close()
    
    def test_seeds(self):
        env1 = sliding_puzzles.make(
            w=self.env_w,
            seed=42
        )
        env1.reset()

        env2 = sliding_puzzles.make(
            w=self.env_w,
            seed=42
        )
        env2.reset()
        self.assertEqual(env1.state.tolist(), env2.state.tolist())

    def test_pickle(self):
        env1 = sliding_puzzles.make(
            w=self.env_w,
        )
        env1.reset()
        with open("test.pkl", "wb") as f:
            pickle.dump(env1, f)

        with open("test.pkl", "rb") as f:
            env2 = pickle.load(f)
        self.assertEqual(env1.state.tolist(), env2.state.tolist())
        env1.close()
        env2.close()
        

if __name__ == "__main__":
    unittest.main()
