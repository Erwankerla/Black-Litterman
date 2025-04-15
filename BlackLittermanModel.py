import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PortfolioOptimization import PortfolioOptimizer


class BLModel:

    def __init__(self, impliedReturns: np.ndarray, tau: float, covMatrix: np.ndarray, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray):
        self.impliedReturns = impliedReturns
        self.P = P
        self.Q = Q
        self.Omega = Omega
        self.covMatrix = covMatrix
        self.tau = tau

        self.viewReturns, self.viewCovMatrix = self.getViewReturns()
        
    
    def getViewReturns(self):
        P, Q = self.P, self.Q

        tauSigma = self.tau * self.covMatrix

        left = tauSigma @ P.T
        mid = np.linalg.inv(self.Omega + P @ tauSigma @ P.T)
        right = Q - P @ self.impliedReturns
        f = np.array(left @ mid @ right)
        v = self.impliedReturns + f.reshape(1, -1)

        if len(self.Omega) == 1:
            invOmega = np.array(1 / self.Omega)
        else:
            invOmega = np.linalg.inv(self.Omega + 1e-20 * np.eye(self.Omega.shape[0]))

        invCov = np.linalg.inv(self.tau * self.covMatrix)
        M = np.linalg.inv(invCov + P.T @ invOmega @ P)

        return v.T, M + self.covMatrix
    

    def plotEfficentFrontier(self, riskAversion, adjustedFrontier:bool):

        if adjustedFrontier is True:
            Returns, Sigma = self.viewReturns.flatten(), self.viewCovMatrix
        else:
            Returns, Sigma = self.impliedReturns.flatten(), self.covMatrix

        ptfOptimizer = PortfolioOptimizer(assetsReturns=Returns, covMatrix=Sigma, riskAversion=riskAversion)

        # ---- Récupérer notre frontière éfficiente du portefeuille ---- #
        optiPtfw, optiR, optiS = ptfOptimizer.getOptimalPortfolio(constraint=True)
        target_returns = np.linspace(np.min(Returns), np.max(Returns), 200)
        portfolio_volatilities, portfolio_returns = [], []
        for target in target_returns:
            weights, ptfReturn, ptfRisk = ptfOptimizer.getOptimalPortfolio(constraint=True, target=target)
            portfolio_returns.append(ptfReturn)
            portfolio_volatilities.append(ptfRisk)

        # ---- Simuler des portefeuilles aléatoires ---- #
        num_portfolios = 50000
        random_returns, random_volatilities = [], []
        for _ in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.viewReturns)), size=1)[0]
            port_return, port_volatility = weights @ Returns.T, np.sqrt(weights @ Sigma @ weights.T)
            random_returns.append(port_return)
            random_volatilities.append(port_volatility)

        # ---- Tracer la frontière efficiente et les points simulés avec Plotly ---- #
        fig = go.Figure()

        # Ajouter les portefeuilles simulés
        fig.add_trace(go.Scatter(
            x=random_volatilities,
            y=random_returns,
            mode='markers',
            name='Portefeuilles Simulés',
            marker=dict(
                color=np.array(random_returns) / np.array(random_volatilities),
                colorscale='Plasma',
                size=2,
                colorbar=dict(title='Sharpe')
            )
        ))

        # Ajouter la frontière efficiente
        fig.add_trace(go.Scatter(
            x=portfolio_volatilities,
            y=target_returns,
            mode='lines',
            name='Frontière Efficiente',
            line=dict(color='black', dash='dash')
        ))

        # Ajouter les actifs individuels
        fig.add_trace(go.Scatter(
            x=np.sqrt(np.diag(Sigma)),
            y=Returns,
            mode='markers+text',
            name='Actifs Individuels',
            marker=dict(color='black', size=10),
            text=["1", "2", "3"],  # Ajustez les étiquettes selon vos actifs
            textposition="top center"
        ))

        # Ajouter le portefeuille optimal
        fig.add_trace(go.Scatter(
            x=[optiS],
            y=[optiR],
            mode='markers',
            name='Portefeuille Optimal',
            marker=dict(color='red', size=15, symbol='star')
        ))

        # Configurer le layout
        fig.update_layout(
            title='Frontière Efficiente avec Black-Litterman et Portefeuilles Simulés',
            xaxis_title='Risque (Volatilité)',
            yaxis_title='Rendement Attendu',
            legend=dict(x=0, y=1),
            #template='plotly_white'
        )

        # Afficher le graphique
        fig.show()


    def plotEfficentFrontierComparison(self, riskAversion):

        
        Returnsadjusted, Sigmaadjusted = self.viewReturns.flatten(), self.viewCovMatrix
        Returns, Sigma = self.impliedReturns.flatten(), self.covMatrix

        ptfOptimizer = PortfolioOptimizer(assetsReturns=Returns, covMatrix=Sigma, riskAversion=riskAversion)
        ptfOptimizerAdjusted = PortfolioOptimizer(assetsReturns=Returnsadjusted, covMatrix=Sigmaadjusted, riskAversion=riskAversion)


        # ---- Récupérer notre frontière éfficiente du portefeuille ---- #
        optiPtfw, optiR, optiS = ptfOptimizer.getOptimalPortfolio(constraint=True)
        target_returns = np.linspace(np.min(Returns), np.max(Returns), 200)
        portfolio_volatilities, portfolio_returns = [], []
        for target in target_returns:
            weights, ptfReturn, ptfRisk = ptfOptimizer.getOptimalPortfolio(constraint=True, target=target)
            portfolio_returns.append(ptfReturn)
            portfolio_volatilities.append(ptfRisk)


         # ---- Récupérer notre frontière éfficiente du portefeuille ajusté ---- #
        optiPtfw, optiR_adjusted, optiS_adjusted = ptfOptimizerAdjusted.getOptimalPortfolio(constraint=True)
        target_returns_adjusted = np.linspace(np.min(Returnsadjusted), np.max(Returnsadjusted), 200)
        portfolio_volatilities_adjusted, portfolio_returns_adjusted = [], []
        for target in target_returns_adjusted:
            weights_adjusted, ptfReturn_adjusted, ptfRisk_adjusted = ptfOptimizerAdjusted.getOptimalPortfolio(constraint=True, target=target)
            portfolio_returns_adjusted.append(ptfReturn_adjusted)
            portfolio_volatilities_adjusted.append(ptfRisk_adjusted)


        dictio = {r:v for r,v in zip(portfolio_returns, portfolio_volatilities)}
        dictio_adjusted = {r:v for r,v in zip(portfolio_returns_adjusted, portfolio_volatilities_adjusted)}

        dictio = dict(sorted(dictio.items(), key=lambda item: item[0]))
        dictio_adjusted = dict(sorted(dictio_adjusted.items(), key=lambda item: item[0]))

        values = list(dictio.values())
        min_index = np.argmin(values)
        new_dict = {k: v for i, (k, v) in enumerate(dictio.items()) if i >= min_index}

        values_adjusted = list(dictio_adjusted.values())
        min_index_adjusted = np.argmin(values_adjusted)
        new_dict_adjusted = {k: v for i, (k, v) in enumerate(dictio_adjusted.items()) if i >= min_index_adjusted}

        portfolio_returns, portfolio_volatilities = list(new_dict.keys()), list(new_dict.values())
        portfolio_returns_adjusted, portfolio_volatilities_adjusted = list(new_dict_adjusted.keys()), list(new_dict_adjusted.values())


        # ---- Tracer les frontières efficientes ---- #
        fig = go.Figure()

        # Ajouter la frontière efficiente
        fig.add_trace(go.Scatter(
            x=portfolio_volatilities,
            y=portfolio_returns,
            mode='lines',
            name='Frontière Efficiente',
            line=dict(color='black')
        ))

        # Ajouter la frontière efficiente ajustée
        fig.add_trace(go.Scatter(
            x=portfolio_volatilities_adjusted,
            y=portfolio_returns_adjusted,
            mode='lines',
            name='Frontière Efficiente ajustée',
            line=dict(color='red')
        ))

        # Ajouter le portefeuille optimal
        fig.add_trace(go.Scatter(
            x=[optiS],
            y=[optiR],
            mode='markers',
            name='Portefeuille Optimal',
            marker=dict(color='black', size=15, symbol='star')
        ))

        # Ajouter le portefeuille optimal ajusté
        fig.add_trace(go.Scatter(
            x=[optiS_adjusted],
            y=[optiR_adjusted],
            mode='markers',
            name='Portefeuille Optimal ajusté',
            marker=dict(color='red', size=15, symbol='star')
        ))

        # Configurer le layout
        fig.update_layout(
            title='Frontière Efficiente',
            xaxis_title='Risque (Volatilité)',
            yaxis_title='Rendement Attendu',
            legend=dict(x=0, y=1, font=dict(size=14)),
            #template='plotly_white'
        )

        # Afficher le graphique
        fig.show()
