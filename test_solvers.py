import unittest
import numpy as np
from recovery_models import forward_euler, rk4_solver, model1_analytical, model3_analytical, H_REST

class TestODESolvers(unittest.TestCase):
    def test_euler_first_order_convergence(self):
        k = 0.0047
        rhs = lambda _t, h: -k * (h - H_REST)
        t2, y2 = forward_euler(rhs, 162.0, 300.0, 2.0)
        t1, y1 = forward_euler(rhs, 162.0, 300.0, 1.0)
        err2 = np.max(np.abs(y2 - model1_analytical(t2, k)))
        err1 = np.max(np.abs(y1 - model1_analytical(t1, k)))
        ratio = err2 / err1
        self.assertTrue(1.95 < ratio < 2.05, f"Expected O(h) ratio ~2.0, got {ratio:.3f}")

    def test_model3_initial_condition_and_asymptote(self):
        t = np.array([0.0, 1e5])
        y = model3_analytical(t, 0.0065, 0.025, h0=162.0)
        self.assertAlmostEqual(y[0], 162.0, places=8)
        self.assertAlmostEqual(y[1], H_REST, places=6)

if __name__ == "__main__":
    unittest.main()
