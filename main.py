from ImpliedReturnsCreator import *
from BlackLittermanInputs import BLInputsCreator, MethodMatrixOmega
from BlackLittermanModel import BLModel
from PortfolioOptimization import PortfolioOptimizer
from PlotFunctions import PlotCreator, SubstractedVector



import pandas as pd


dataBench = pd.read_excel("CAC.xlsx")
dataBench.set_index("Date", drop=True, inplace=True)

dataAssets = pd.read_excel("Assets.xlsx")
dataAssets.set_index("Date", drop=True, inplace=True)


riskFreeRate = 0.0


impliedReturnsCreator = ImpliedReturnsCreator(riskFreeRate, dataBench, dataAssets, typeCovarianceMatrix=TypeCovarianceMatrix.CLASSIC,
                                               tauCalibrationType=TauCalibrationType.LIKELIHOOD)

marketWeights = impliedReturnsCreator.marketWeights
marketReturn = impliedReturnsCreator.marketReturn
sigmaMarket = impliedReturnsCreator.sigmaMarket

riskAversion = impliedReturnsCreator.riskAversion

covarianceMatrix = impliedReturnsCreator.covarianceMatrix

impliedReturns = impliedReturnsCreator.impliedReturns

TAU = impliedReturnsCreator.tau


blinputs = BLInputsCreator(covarianceMatrix, tau= TAU, numAssets=len(impliedReturns), methodOmega=MethodMatrixOmega.IDZOREK)
P, Q, probaOpinion = blinputs.P, blinputs.Q, blinputs.probaOpinion
Omega = blinputs.getMatrix_Omega(impliedReturns=impliedReturns, marketWeights=marketWeights, riskAversion=riskAversion)

blmodel = BLModel(impliedReturns, tau= TAU, covMatrix=covarianceMatrix, P=P, Q=Q, Omega=Omega)
adjustedReturns = blmodel.viewReturns
adjustedCovMatrix = blmodel.viewCovMatrix

print(impliedReturns, adjustedReturns.T, impliedReturns - adjustedReturns.T)


ptfOptimizer = PortfolioOptimizer(adjustedReturns, adjustedCovMatrix, riskAversion)
optimalWeights, ptfReturn, ptfRisk = ptfOptimizer.getOptimalPortfolio(constraint=True)
print(marketWeights, optimalWeights, ptfReturn, ptfRisk)

blmodel.plotEfficentFrontier(riskAversion, adjustedFrontier=True)
#blmodel.plotEfficentFrontier(riskAversion=riskAversion, adjustedFrontier=False)
blmodel.plotEfficentFrontierComparison(riskAversion=riskAversion)





#tracer la capital market line. prend rendement marché (le bench ? somme pondéré des rendements des assets ?)

plotFunctions = PlotCreator(dataBench=dataBench, dataAssets=dataAssets, riskFreeRate=riskFreeRate)

#plotFunctions.plotTauEvolution(substractedVector=SubstractedVector.Pi, methodOmega=MethodMatrixOmega.IDZOREK)
#plotFunctions.plotTauEvolutionVar(methodOmega=MethodMatrixOmega.IDZOREK)
#plotFunctions.plotTauEvolution(substractedVector=SubstractedVector.Q, methodOmega=MethodMatrixOmega.IDZOREK)

#plotFunctions.plotDistanceEvolution(normalized=False, methodOmega=MethodMatrixOmega.WALTERS)
#plotFunctions.plotDistanceEvolution(normalized=True, methodOmega=MethodMatrixOmega.WALTERS)

#plotFunctions.plotWeightBar(methodOmega=MethodMatrixOmega.IDZOREK)

#plotFunctions.plotDistanceEvolutionSurface3D(normalized=True, methodOmega=MethodMatrixOmega.WALTERS)
#plotFunctions.plotDistanceEvolutionAllSurface3D(normalized=True, methodOmega=MethodMatrixOmega.WALTERS)
