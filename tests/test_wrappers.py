import pickle
import unittest
import sliding_puzzles


class TestSlidingEnvWrappers(unittest.TestCase):
    def setUp(self):
        self.env_w = 4

    def test_pickle(self):
        env1 = sliding_puzzles.make(
            w=self.env_w,
            variation="image",
            image_folder="test",
            image_pool_size=2,
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
