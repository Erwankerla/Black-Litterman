from ImpliedReturnsCreator import *
from BlackLittermanInputs import BLInputsCreator, MethodMatrixOmega
from BlackLittermanModel import BLModel
from PortfolioOptimization import PortfolioOptimizer
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from enum import Enum
import matplotlib.pyplot as plt



class SubstractedVector(Enum):
    Pi = "Pi"
    Q = "Q"



class PlotCreator:

    def __init__(self, dataBench, dataAssets, riskFreeRate):
        self.dataBench = dataBench
        self.dataAssets = dataAssets
        self.riskFreeRate = riskFreeRate

            
    def plotTauEvolution(self, substractedVector:SubstractedVector, methodOmega:MethodMatrixOmega, P=None, Q=None, probaOpinion=None):
        
        dataBench, dataAssets, riskFreeRate = self.dataBench, self.dataAssets, self.riskFreeRate

        if P is None and Q is None and probaOpinion is None:
            P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            Q = np.array([0.01, 0.05, 0.042], dtype=float)
            probaOpinion = np.array([0.8, 0.9, 0.5], dtype=float)

        impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets, typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                                      tauCalibrationType=TauCalibrationType.FREQ)

        marketWeights = impliedReturnsCreator.marketWeights
        riskAversion = impliedReturnsCreator.riskAversion
        covarianceMatrix = impliedReturnsCreator.covarianceMatrix
        impliedReturns = impliedReturnsCreator.impliedReturns

        listeTau = np.linspace(0.015, 30, 100)

        differences_list = []
        differences_test_list = []

        for tau in listeTau:
            blinputs = BLInputsCreator(covMatrix=covarianceMatrix, tau=tau, numAssets=len(impliedReturns), methodOmega=methodOmega, P=P, Q=Q, probaOpinion=probaOpinion)
            Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
            testOmega = blinputs.getMatrix_OmegaBis(tilt=False)

            # Modèle Idzorek
            blmodel = BLModel(impliedReturns, tau=tau, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
            adjustedReturns = blmodel.viewReturns.flatten()
            # Modèle test
            blmodelTest = BLModel(impliedReturns, tau=tau, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=testOmega)
            adjustedReturnsTest = blmodelTest.viewReturns.flatten() 

            vector = impliedReturns if substractedVector is SubstractedVector.Pi else Q
            differences = np.abs(adjustedReturns - vector.flatten())
            differences_test = np.abs(adjustedReturnsTest - vector.flatten())

            differences_list.append(differences)
            differences_test_list.append(differences_test)

        df_differences = pd.DataFrame(differences_list, columns=[f'Actif {i+1}' for i in range(len(impliedReturns))])
        df_differences['Tau'] = listeTau

        df_differences_test = pd.DataFrame(differences_test_list, columns=[f'Actif {i+1}' for i in range(len(impliedReturns))])
        df_differences_test['Tau'] = listeTau

        colors = ['salmon', 'purple', 'blue', 'orange', 'red', 'green', 'cyan', 'brown', 'pink', 'gray']

        fig = go.Figure()

        # Ajouter les courbes pour le modèle Idzorek
        for i in range(len(impliedReturns)):
            fig.add_trace(go.Scatter(
                x=df_differences['Tau'],
                y=df_differences[f'Actif {i+1}'],
                mode='lines',
                name=f'Actif {i+1} (Idzorek)',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))

        # Ajouter les courbes pour le modèle classique
        for i in range(len(impliedReturns)):
            fig.add_trace(go.Scatter(
                x=df_differences_test['Tau'],
                y=df_differences_test[f'Actif {i+1}'],
                mode='lines',
                name=f'Actif {i+1} (Classique)',
                line=dict(color=colors[i % len(colors)])
            ))

        fig.update_layout(
            title="Différences des Rendements Ajustés par Rapport aux Rendements Implicites selon Tau",
            title_x=0.5, 
            xaxis_title="Tau",
            yaxis_title="Différence des Rendements Ajustés",
            legend_title="Légende",
            legend=dict(
                x=1, 
                y=0.5,
                traceorder="normal",
                orientation="v"
            ),
            #template='plotly_white'
        )

        fig.show()

    

    def plotTauEvolutionVar(self, methodOmega:MethodMatrixOmega, P=None, Q=None, probaOpinion=None):
        
        dataBench, dataAssets, riskFreeRate = self.dataBench, self.dataAssets, self.riskFreeRate

        if P is None and Q is None and probaOpinion is None:
            P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            Q = np.array([0.01, 0.05, 0.042], dtype=float)
            probaOpinion = np.array([0.8, 0.9, 0.5], dtype=float)


        impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets, typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                                      tauCalibrationType=TauCalibrationType.FREQ)

        marketWeights = impliedReturnsCreator.marketWeights
        riskAversion = impliedReturnsCreator.riskAversion
        covarianceMatrix = impliedReturnsCreator.covarianceMatrix
        impliedReturns = impliedReturnsCreator.impliedReturns

        listeTau = np.linspace(0.015, 1, 30)

        var_list = []
        var_test_list = []

        for tau in listeTau:
            blinputs = BLInputsCreator(covMatrix=covarianceMatrix, tau=tau, numAssets=len(impliedReturns), methodOmega=methodOmega, P=P, Q=Q, probaOpinion=probaOpinion)
            Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
            testOmega = blinputs.getMatrix_OmegaBis(tilt=False)

            # Modèle Idzorek
            blmodel = BLModel(impliedReturns, tau=tau, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
            adjustedCovMatrix = blmodel.viewCovMatrix

            # Modèle test
            blmodelTest = BLModel(impliedReturns, tau=tau, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=testOmega)
            adjustedCovMatrixTest = blmodelTest.viewCovMatrix

            var_list.append(np.diag(adjustedCovMatrix))
            var_test_list.append(np.diag(adjustedCovMatrixTest))

        df_differences = pd.DataFrame(var_list, columns=[f'Actif {i+1}' for i in range(len(impliedReturns))])
        df_differences['Tau'] = listeTau

        df_differences_test = pd.DataFrame(var_test_list, columns=[f'Actif {i+1}' for i in range(len(impliedReturns))])
        df_differences_test['Tau'] = listeTau

        colors = ['salmon', 'purple', 'blue', 'orange', 'red', 'green', 'cyan', 'brown', 'pink', 'gray']

        fig = go.Figure()

        # Ajouter les courbes pour le modèle Idzorek
        for i in range(len(impliedReturns)):
            fig.add_trace(go.Scatter(
                x=df_differences['Tau'],
                y=df_differences[f'Actif {i+1}'],
                mode='lines',
                name=f'Actif {i+1} (Idzorek)',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))

        # Ajouter les courbes pour le modèle classique
        for i in range(len(impliedReturns)):
            fig.add_trace(go.Scatter(
                x=df_differences_test['Tau'],
                y=df_differences_test[f'Actif {i+1}'],
                mode='lines',
                name=f'Actif {i+1} (Classique)',
                line=dict(color=colors[i % len(colors)])
            ))

        fig.update_layout(
            title="Evolution des variances selon Tau",
            title_x=0.5, 
            xaxis_title="Tau",
            yaxis_title="Variance des Rendements Ajustés",
            legend_title="Légende",
            legend=dict(
                x=1, 
                y=0.5,
                traceorder="normal",
                orientation="v"
            ),
            #template='plotly_white'
        )

        fig.show()




    def plotDistanceEvolution(self, normalized:bool, methodOmega:MethodMatrixOmega, P=None, Q=None):

        dataBench, dataAssets, riskFreeRate = self.dataBench, self.dataAssets, self.riskFreeRate


        if P is None and Q is None:
            P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            Q = np.array([0.074, 0.05, 0.042])
            
        impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets, typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                                      tauCalibrationType=TauCalibrationType.FREQ)

        marketWeights = impliedReturnsCreator.marketWeights
        riskAversion = impliedReturnsCreator.riskAversion
        covarianceMatrix = impliedReturnsCreator.covarianceMatrix
        impliedReturns = impliedReturnsCreator.impliedReturns

        probas = np.linspace(0.001, 1, 100)
        differences = np.zeros((len(probas), len(marketWeights)))

        for i, proba in enumerate(probas):
            blinputs = BLInputsCreator(covMatrix=covarianceMatrix, tau=0.25, numAssets=len(impliedReturns), 
                                    methodOmega=methodOmega, P=P, Q=Q, probaOpinion=np.array([proba] * len(impliedReturns)))
            Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
            
            blmodel = BLModel(impliedReturns, tau=0.25, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
            adjustedReturns = blmodel.viewReturns.flatten()

            ptfOpti = PortfolioOptimizer(adjustedReturns, covarianceMatrix, riskAversion)
            w = ptfOpti.getOptimalPortfolio(constraint=False)[0]
            
            differences[i] = np.abs(marketWeights.flatten() - w.flatten())

        if normalized is True:
            max_differences = np.max(differences, axis=0)
            differences = differences / max_differences

        colors = ['salmon', 'purple', 'blue', 'orange', 'red', 'green', 'cyan', 'brown', 'pink', 'gray']

        # Plotting using Plotly
        fig = go.Figure()

        for j in range(len(marketWeights)):
            fig.add_trace(go.Scatter(
                x=probas,
                y=differences[:, j],
                mode='lines',
                name=f'Actif {j + 1}',
                line=dict(color=colors[j % len(colors)])  # Cycle through colors
            ))

        fig.update_layout(
            title = f"Evolution of {'Normalized ' if normalized else ''}Differences between Market Weights and Adjusted Weights",
            xaxis_title = 'Probability',
            yaxis_title = f"{'Normalized ' if normalized else ''}Difference (Market Weight - Adjusted Weight)",
            yaxis=dict(showgrid=True),
            xaxis=dict(showgrid=True),
            showlegend=True,
            legend=dict(
                x=1, 
                y=0.5,
                traceorder="normal",
                orientation="v"
            ),
            width=600,
            height=600 
        )

        if normalized is True:
            fig.add_hline(y=1, line_color='black', line_dash='dash', annotation_text='y=1', annotation_position='top right')  # Horizontal line at y=1
        fig.show()



    def plotDistanceEvolutionSurface3D(self, normalized:bool, methodOmega:MethodMatrixOmega, P=None, Q=None):

        dataBench, dataAssets, riskFreeRate = self.dataBench, self.dataAssets, self.riskFreeRate

        if P is None and Q is None:
            P = np.array([[1, 0], [0, 1]])
            Q = np.array([0.03, 0.08])
            
        impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets.iloc[:, :2], typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                                      tauCalibrationType=TauCalibrationType.FREQ)

        marketWeights = impliedReturnsCreator.marketWeights
        riskAversion = impliedReturnsCreator.riskAversion
        covarianceMatrix = impliedReturnsCreator.covarianceMatrix
        impliedReturns = impliedReturnsCreator.impliedReturns

        probas = np.linspace(0.001, 1, 50)
        differences = np.zeros((len(probas), len(probas)))

        for i, proba1 in enumerate(probas):
            for j, proba2 in enumerate(probas):

                blinputs = BLInputsCreator(covMatrix=covarianceMatrix, tau=0.25, numAssets=len(impliedReturns), 
                                        methodOmega=methodOmega, P=P, Q=Q, probaOpinion=np.array([proba1, proba2]))
                Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
                
                blmodel = BLModel(impliedReturns, tau=0.25, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
                adjustedReturns = blmodel.viewReturns.flatten()

                ptfOpti = PortfolioOptimizer(adjustedReturns, covarianceMatrix, riskAversion)
                w = ptfOpti.getOptimalPortfolio(constraint=False)[0]
                
                differences[i, j] = np.linalg.norm(marketWeights.flatten() - w.flatten())

        if normalized is True:
            maxdiff = np.max(differences)
            differences /= maxdiff
            
        fig = go.Figure(data=[
            go.Surface(
                z=differences, 
                x=probas,
                y=probas,
                colorscale='Jet', 
                colorbar=dict(title='Differences'),
                contours=dict(
                    z=dict(show=True, size=5)
                )
            )
        ])

        fig.update_layout(
            title='Surface 3D des Différences',
            scene=dict(
                xaxis_title='Proba 1',
                yaxis_title='Proba 2',
                zaxis_title='Differences'
            )
        )

        fig.show()



    def plotDistanceEvolutionAllSurface3D(self, normalized: bool, methodOmega: MethodMatrixOmega, P=None, Q=None):
        dataBench, dataAssets, riskFreeRate = self.dataBench, self.dataAssets, self.riskFreeRate

        if P is None and Q is None:
            P = np.array([[1, 0], [0, 1]])
            Q = np.array([0.03, 0.08])
            
        impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets.iloc[:, :2], typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                                      tauCalibrationType=TauCalibrationType.FREQ)

        marketWeights = impliedReturnsCreator.marketWeights
        riskAversion = impliedReturnsCreator.riskAversion
        covarianceMatrix = impliedReturnsCreator.covarianceMatrix
        impliedReturns = impliedReturnsCreator.impliedReturns

        probas = np.linspace(0.001, 1, 50)
        differences = np.zeros((len(probas), len(probas), len(marketWeights)))

        for i, proba1 in enumerate(probas):
            for j, proba2 in enumerate(probas):
                blinputs = BLInputsCreator(covMatrix=covarianceMatrix, tau=0.25, numAssets=len(impliedReturns), 
                                            methodOmega=methodOmega, P=P, Q=Q, probaOpinion=np.array([proba1, proba2]))
                Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
                
                blmodel = BLModel(impliedReturns, tau=0.25, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
                adjustedReturns = blmodel.viewReturns.flatten()

                ptfOpti = PortfolioOptimizer(adjustedReturns, covarianceMatrix, riskAversion)
                w = ptfOpti.getOptimalPortfolio(constraint=False)[0]
                
                differences[i, j] = np.abs(marketWeights.flatten() - w.flatten())

        fig = go.Figure()

        for k in range(len(impliedReturns)):

            if normalized is True:
                maxdiff = np.max(differences[:, :, k])
                differences[:, :, k] = differences[:, :, k] / maxdiff

            fig.add_trace(go.Surface(
                z=differences[:, :, k],  # Utilisation des différences pour l'actif k
                x=probas,
                y=probas,
                colorscale='Viridis', 
                showscale=True,
                name=f'Asset {k+1}', 
                opacity=0.9,
                contours=dict(
                    z=dict(show=True, size=5)
                )
            ))

        fig.update_layout(
            title='Surface 3D des Différences pour chaque Actif',
            scene=dict(
                xaxis_title='Proba 1',
                yaxis_title='Proba 2',
                zaxis_title='Differences'
            )
        )

        fig.show()



    def plotWeightBar(self, methodOmega:MethodMatrixOmega, P=None, Q=None, probaOpinion=None):

        dataBench, dataAssets, riskFreeRate = self.dataBench, self.dataAssets, self.riskFreeRate


        if P is None and Q is None and probaOpinion is None:
            P = np.array([[1, 0, 0], [0, 1, 0]])
            Q = np.array([0.084, 0.05])
            probaOpinion = np.array([0.7, 0.7])
            
        impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets, typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                                      tauCalibrationType=TauCalibrationType.FREQ)

        marketWeights = impliedReturnsCreator.marketWeights
        riskAversion = impliedReturnsCreator.riskAversion
        covarianceMatrix = impliedReturnsCreator.covarianceMatrix
        impliedReturns = impliedReturnsCreator.impliedReturns


        blinputs100 = BLInputsCreator(covMatrix=covarianceMatrix, tau=0.25, numAssets=len(impliedReturns), methodOmega=methodOmega, 
                                   P=P, Q=Q, probaOpinion=np.array([1.0] * len(Q)))
        Omega100 = blinputs100.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
        
        blmodel100 = BLModel(impliedReturns, tau=0.25, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega100)
        adjustedReturns100 = blmodel100.viewReturns.flatten()
        
        ptfOpti100 = PortfolioOptimizer(adjustedReturns100, covarianceMatrix, riskAversion)
        w100 = ptfOpti100.getOptimalPortfolio(constraint=False)[0]
        
        blinputs = BLInputsCreator(covMatrix=covarianceMatrix, tau=0.25, numAssets=len(impliedReturns), methodOmega=methodOmega, 
                                   P=P, Q=Q, probaOpinion=probaOpinion)
        Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)
        
        blmodel = BLModel(impliedReturns, tau=0.25, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
        adjustedReturns = blmodel.viewReturns.flatten()
        
        ptfOpti = PortfolioOptimizer(adjustedReturns, covarianceMatrix, riskAversion)
        w = ptfOpti.getOptimalPortfolio(constraint=False)[0]
        

        weights = np.array([marketWeights.flatten(), w.flatten(), w100.flatten()])
        labels = ['Market Weights', 'Adjusted Weights', 'Adjusted Weights 100']

        colors = ['salmon', 'purple', 'blue', 'orange', 'red', 'green', 'cyan', 'brown', 'pink', 'gray']

        num_assets = weights.shape[1]
        fig = go.Figure()

        for i in range(weights.shape[0]):
            fig.add_trace(go.Bar(
                x=[f'Asset {j+1}' for j in range(num_assets)],
                y=weights[i],
                name=labels[i],
                marker_color=colors[i],
                opacity=0.8, 
                hoverinfo='y+name'  
            ))

        fig.update_layout(
            title='Market Weights, Adjusted Weights, and Adjusted Weights 100',
            yaxis_title='Weights',
            barmode='group',
            scene=dict(
                xaxis=dict(title='Assets'),
                yaxis=dict(title='Weights'),
                zaxis=dict(title=''),
            ),
            legend=dict(
                x=1, 
                y=0.5,
                traceorder="normal",
                orientation="v"
            ),
            #template='plotly_white',
            width=600,
            height=600 
            
        )

        fig.show()