import pandas as pd
import numpy as np
from enum import Enum
import math
from scipy.optimize import minimize

class TypeCovarianceMatrix(Enum):
    CLASSIC = "classic"
    EXPONENTIAL = "expo"

class TauCalibrationType(Enum):
    FREQ = 'freq'
    LIKELIHOOD = 'likelihood'


class ImpliedReturnsCreator:
    """Cet Classe permet de calculer les rendements implicites. Elle permet aussi d'avoir accès à:
                   - marketWeights
                   - riskAversion
                   - covarianceMatrix
                   - impliedRetruns 
    """

    def __init__(self, riskFreeRate: float, BenchMark: pd.DataFrame, Assets: pd.DataFrame, typeCovarianceMatrix: TypeCovarianceMatrix, tauCalibrationType: TauCalibrationType):
        self.riskFreeRate = riskFreeRate
        self.BenchMark = BenchMark
        self.Assets = Assets
        self.typeCovarianceMatrix = typeCovarianceMatrix
        self.tauCalibrationType = tauCalibrationType

        self.marketReturn = None
        self.sigmaMarket = None

        Assets.reset_index(drop=True, inplace=True)
        BenchMark.reset_index(drop=True, inplace=True)

        Assets = Assets.iloc[::-1].reset_index(drop=True)
        BenchMark = BenchMark.iloc[::-1].reset_index(drop=True)    

        returnsBench = BenchMark.pct_change().dropna()
        returnsAssets = Assets.pct_change().dropna()

        self.marketWeights = self.getMarketWeights(Assets)
        self.covarianceMatrix = self.getCovarianceMatrix(returnsAssets)
        self.riskAversion = self.getRiskAversion(riskFreeRate, returnsBench)
        
        self.impliedReturns = np.array(self.riskAversion * self.covarianceMatrix @ self.marketWeights.T)

        #rajouter la possibilité de choisir notre fréquence (52, 252, 12, 1)
        self.tau = 252 / len(BenchMark) if tauCalibrationType is TauCalibrationType.FREQ else self.calibrationTau()
 
    

    def calibrationTau(self):
        impliedReturns, dataAssets, covarianceMatrix = self.impliedReturns, self.Assets, self.covarianceMatrix
        # Calcule des rendements annualisés des actifs
        assetsReturns = (1 + dataAssets.pct_change().mean()) ** 252 - 1
        n = len(assetsReturns)

        M = assetsReturns - impliedReturns
        inv_cov = np.linalg.inv(covarianceMatrix + 1e-20 * np.eye(covarianceMatrix.shape[0]))
        log_det_cov = math.log(np.linalg.det(covarianceMatrix) + 1e-20)  # Ajout d'un petit nombre pour éviter les problèmes de log(0)

        def f(k):
            # Calcul de la fonction de log-vraisemblance
            return -0.5 * (math.log(k) - log_det_cov - n * math.log(2 * math.pi) - k * (M.T @ inv_cov @ M))

        k0 = 2
        constraints = [{'type': 'eq', 'fun': lambda k: k - int(k)},  # k doit être entier
                       {'type': 'ineq', 'fun': lambda k: k - 1}] # k doit être > 1
        
        results = minimize(f, k0, constraints=constraints, method='SLSQP')

        if not results.success:
            raise RuntimeError("La minimisation a échoué : " + results.message)
        
        return results.x[0]


    def getMarketWeights(self, Assets: pd.DataFrame):
        """Compute the weights respectively to the market value of each assets"""

        lastsPrices = Assets.iloc[-1, :].values
        sum = np.sum(lastsPrices)

        return np.array(lastsPrices / sum)


    def getRiskAversion(self, riskFreeRate, returnsBench: pd.DataFrame):
        """This risk aversion coeff is based on the CAPM and the following formula is obtain after linear algebra calculus"""

        sigmaMarket = np.sqrt(252)*np.std(returnsBench)        
        meanMarket = returnsBench.mean()
        marketReturn = (1 + meanMarket) ** 252 - 1   #annualized market returns
        self.marketReturn = marketReturn
        self.sigmaMarket = sigmaMarket

        return float((marketReturn - riskFreeRate) / (sigmaMarket * sigmaMarket))


    def getCovarianceMatrix(self, returnsAssets: pd.DataFrame):

        if self.typeCovarianceMatrix is TypeCovarianceMatrix.CLASSIC:
            return np.array(252 * returnsAssets.cov())
        
        elif self.typeCovarianceMatrix is TypeCovarianceMatrix.EXPONENTIAL:   #Revoir cettte méthode
            
            """Based on scientific studies, 0.94 is a good value for taking account both present value and past value with significant impact on the covariance"""
            LAMBDA = 0.94

            cov_matrix = np.zeros((returnsAssets.shape[1], returnsAssets.shape[1]))

            # Calculate weight
            weights = np.exp(-LAMBDA * returnsAssets)  # Ensure this results in the correct shape
            weights /= weights.sum(axis=0)  # Normalize weights for each asset

            for i in range(returnsAssets.shape[1]):
                for j in range(returnsAssets.shape[1]):
                    mean_i = returnsAssets.iloc[:, i].mean()
                    mean_j = returnsAssets.iloc[:, j].mean()
                    
                    deviations_product = (returnsAssets.iloc[:, i] - mean_i) * (returnsAssets.iloc[:, j] - mean_j)
                    
                    weighted_sum = np.sum(weights * deviations_product)

                    cov_matrix[i, j] = 252 * weighted_sum

            return np.array(cov_matrix)