import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:

    def __init__(self, assetsReturns: np.ndarray, covMatrix: np.ndarray, riskAversion:float):
        self.assetsReturns = assetsReturns
        self.covMatrix = covMatrix
        self.riskAversion = riskAversion


    def getOptimalPortfolio(self, constraint: bool, target=None): 
        viewReturns = self.assetsReturns 
        viewCovMatrix = self.covMatrix # shape (3, 3)

        n = len(self.assetsReturns) 

        if constraint is True:
            def objective(w):
                w = np.asarray(w)
                return -(w @ viewReturns - 0.5 * self.riskAversion * w @ viewCovMatrix @ w)

            if target is not None:
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, # Les poids doivent sommer à 1
                    {'type': 'eq', 'fun': lambda w: w @ viewReturns - target} # Rendement cible
                ]
            else:
                constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            bounds = tuple((0, 1) for _ in range(n))

            initial_weights = np.ones(n) / n # shape (3,)

            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = np.array(result.x) 
                expected_return = optimal_weights @ viewReturns
                portfolio_std = np.sqrt(optimal_weights @ viewCovMatrix @ optimal_weights.T)

                return optimal_weights, expected_return, portfolio_std
            else:
                raise ValueError("Optimization failed: " + result.message)
        
        else:
             invCovRisk = np.linalg.inv(self.riskAversion * self.covMatrix)
             return invCovRisk @ self.assetsReturns.T, []
