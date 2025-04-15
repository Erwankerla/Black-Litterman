import numpy as np
from scipy.optimize import minimize
from enum import Enum
from PortfolioOptimization import PortfolioOptimizer



class MethodMatrixOmega(Enum):
    IDZOREK = "Idzorek"
    WALTERS = "Walter"


class BLInputsCreator:
    """Cette classe permet de créer les matrices liées aux opinions de l'investisseur:
                - P : matrice qui contient les ordres de notre opinion
                - Q : matrice dans laquelle nous mettons les prévisions des performances des opinions
                - Omega : matrice d'incertitude sur les opinions (plus compliqué à déterminer)"""
    
    def __init__(self, covMatrix: np.ndarray, tau: float, numAssets: int, methodOmega: MethodMatrixOmega, P:np.ndarray = None, Q:np.ndarray = None, probaOpinion:np.ndarray = None):
        self.n = numAssets
        self.tau = tau
        self.covMatrix = covMatrix
        self.methodOmega = methodOmega

        if P is None and Q is None and probaOpinion is None:
            self.P, self.Q, self.probaOpinion = self.getMatrix_P_Q_ProbaOpinion()
        else:
            self.P, self.Q, self.probaOpinion = P, Q, probaOpinion


    def getMatrix_P_Q_ProbaOpinion(self):
        continu = True
        P = []
        Q = []
        probaOpinion = []

        while continu:
            opinion = []
            for i in range(self.n):
                weight = float(input(f"Entrez le poids de l'actif {i + 1} (en tant que nombre) : "))
                opinion.append(weight)
            
            forecast = float(input("Entrez votre prévision de performance pour cette opinion : "))
            
            P.append(opinion)
            Q.append(forecast)

            probability = float(input("Entrez votre confiance pour cette opinion (entre 0 et 1) : "))
            probaOpinion.append(probability)

            user_input = input("Voulez-vous ajouter une nouvelle opinion ? (True / False) : ")
            continu = user_input.lower() == 'true'
        
        P = np.array(P)
        Q = np.array(Q)
        probaOpinion = np.array(probaOpinion)

        return P, Q, probaOpinion
    

    def getMatrix_OmegaBis(self, tilt:bool):
        """Calculate the Omega matrix based on the opinions and covariance."""

        if tilt is False:
            return np.diag([1 * (self.P[i, :] @ self.covMatrix @ self.P[i, :].T).item() for i in range(len(self.P))])

        else:
            raise ValueError("please for this funstion enter False")
        
    
    def getMatrix_Omega(self, impliedReturns:np.ndarray, marketWeights:np.ndarray, riskAversion:float):
        """Calculate the Omega matrix based on the opinions and covariance."""

        P, Q, covMatrix = self.P, self.Q, self.covMatrix
        marketWeights = np.array(marketWeights.flatten())
        impliedReturns = np.array(impliedReturns.flatten())
        
        tauSigma = self.tau * covMatrix
        invCovRisk = np.linalg.inv(riskAversion * covMatrix)
        invCovTau = np.linalg.inv(tauSigma)

        optimalOmega = []
        new_W, Ds = [], []
        for i in range(len(P)):
            p, q = np.array(P[i, :].flatten()), Q[i]

            inv = float(np.linalg.inv(p.reshape(1, -1) @ tauSigma @ p.reshape(-1, 1)))
            left = np.array((tauSigma @ p.T * inv).flatten())
            right = float((q - p @ impliedReturns.T))
            R_100 = impliedReturns + left * right

            ptfOptimizer = PortfolioOptimizer(assetsReturns= R_100, covMatrix= covMatrix, riskAversion=riskAversion)
            w_k_100 = ptfOptimizer.getOptimalPortfolio(constraint=False)[0]

            D_k_100 = w_k_100 - marketWeights

            C_k = self.probaOpinion[i] 
            C = np.where(p != 0, float(C_k), p)

            tilt_k = D_k_100 * C

            new_w = marketWeights + tilt_k

            new_W.append(new_w)
            Ds.append(D_k_100)


        if self.methodOmega is MethodMatrixOmega.IDZOREK:
            
            for i in range(len(new_W)):
                p, q = P[i, :].reshape(-1, 1), np.array(Q[i]).reshape(-1, 1)
                new_w = np.array(new_W[i]).reshape(-1, 1)

                def optimizationFunction(omega_k):
                    omega_k = np.atleast_2d(omega_k)

                    left = invCovRisk @ np.linalg.inv(invCovTau + p @ np.linalg.inv(omega_k + 1e-8 * np.eye(omega_k.shape[0])) @ p.T)
                    right = np.array((invCovTau @ impliedReturns.reshape(-1, 1) + p @ np.linalg.inv(omega_k + 1e-8 * np.eye(omega_k.shape[0])) @ q))

                    w_k = left @ right.reshape(-1, 1)
                    return np.linalg.norm(np.array(new_w) - np.array(w_k))**2
                
                #contrainte de stricte positivité
                constraints = [{'type': 'ineq', 'fun': lambda omega_k: omega_k - 10e-30}]
                initialOmega_k = 0.05
                optimalOmega_k = minimize(optimizationFunction, initialOmega_k, method='SLSQP', constraints=constraints, bounds=[(0, None)])

                optimalOmega.append(optimalOmega_k.x)
            
            flattened_optimalOmega = [float(omega) for omega in optimalOmega]
            return np.diag(flattened_optimalOmega)

        else:
            alphas = []
            for i in range(len(new_W)): # pas logique de faire ca car alpha est un vecteur 
                new_w, D = new_W[i], Ds[i]
                Iconfiance = np.linalg.norm(new_w - marketWeights) / np.linalg.norm(D)
                alpha = 1 / Iconfiance - 1 
                alphas.append(alpha)
            
            return np.diag([alphas[i] * (self.P[i, :] @ tauSigma @ self.P[i, :].T).item() for i in range(len(self.P))])

