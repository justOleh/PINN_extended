import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j0, y0
from scipy.optimize import brentq

from typing import Any


class ConcentrationSolution:
    """
        Class computes concentration of some substance in cylindrical object
        provided boundary and initial conditions.  
    """
    def __init__(self, r1: float, r2: float, c1: float, c2: float, D: float) -> None:
        self.r1 = r1
        self.r2 = r2
        self.c1 = c1
        self.c2 = c2
        self.D = D
        self.beta_n = self.find_roots(beta_start=0.001, beta_finish=50, step=0.001)

        # check correctness of the roots
        if not np.allclose(self.U0(self.beta_n, r1), 0, rtol=1e-05, atol=6*1e-4):
            raise ValueError("Roots of equation U0 = 0 are not close enough. Try different parameters.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        r, t = args
        return self.calc_value(r, t)
 
    def find_root(self, beta_interval: tuple[float, float]):
        """
            Finds root of U0 in interval beta_interval. 
        """
        a, b = beta_interval
        U0_partial = lambda beta: self.U0(beta, self.r1)
        root = brentq(U0_partial, a, b)
        return root

    def find_beta_intervals(self, beta_start: float = 0.01, beta_finish: float = 100, step: float = 0.0001):

        def find_sign_changes(values):
            sign_changes = []
            for i in range(1, len(values)):
                if (values[i] >= 0 and values[i - 1] < 0) or (values[i] < 0 and values[i - 1] >= 0):
                    sign_changes.append((i - 1, i))
            return sign_changes

        beta_n = np.arange(beta_start, beta_finish, step)
        values = self.U0(r=self.r1, beta=beta_n)
        indexes = find_sign_changes(values) 

        return beta_n[indexes]

    def find_roots(self, beta_start, beta_finish, step) -> np.ndarray:
        beta_intervals = self.find_beta_intervals(beta_start, beta_finish, step)
        roots = [self.find_root(beta_interval) for beta_interval in beta_intervals]
        return np.array(roots)

    def U0(self, beta, r):
        return j0(beta*r)*y0(beta*self.r2) - j0(beta*self.r2)*y0(beta*r)

    def calc_second_term(self, r, t) -> float:
        j0_r2 = j0(self.beta_n*self.r2)
        j0_r1 = j0(self.beta_n*self.r1)
        U0_vals = self.U0(self.beta_n, r)

        numerator = self.c1*j0_r2*j0_r1*U0_vals*np.exp(-self.beta_n**2*self.D*t)
        denominator = j0_r1**2-j0_r2**2
        series_value = np.sum(numerator/denominator)

        return np.pi*series_value

    def calc_first_term(self, r):
        numerator = np.log(self.r2 / r)
        denominator = np.log(self.r2 / self.r1)
        result = self.c1 * numerator / denominator
        return result

    def calc_value(self, r, t):
        first_term = self.calc_first_term(r)
        second_term = self.calc_second_term(r, t)

        return float(first_term + second_term)
    

if __name__ == "__main__":
    r1, r2, c1, c2, c0, D = 598, 610, 4, 0, 0,  3.2*1e-3

    t_start = 1
    t_finish = 25*1e3

    C = ConcentrationSolution(r1=r1, r2=r2, c1=c1, c2=c2, D=D)

    print(C(598, 0), C(599, 0), C(600, 0), C(601, 0), C(601, 0), C(610, 0))
    print(C(598, 1000), C(599, 1000), C(600, 1000), C(601, 1000), C(601, 1000), C(610, 1000))
    print(C(598, t_finish), C(599, t_finish), C(600, t_finish), C(601, t_finish), C(601, t_finish), C(610, t_finish))
