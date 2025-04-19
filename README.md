# Modèle de Black-Litterman

Ce projet contient un travail de recherche approfondi sur le modèle de Black-Litterman en finance, incluant ses variantes et extensions.

📄 Un document complet expliquant le modèle, ses fondements théoriques et ses déclinaisons est disponible dans le dossier `Documents/`.

📚 Tous les articles scientifiques qui ont servi de base à la rédaction de ce papier sont regroupés dans le dossier `Articles/`.

💻 Le code Python associé, ainsi que les données utilisées pour illustrer le modèle, se trouvent dans le dossier `Implémentation/`:

- `ImpliedReturnsCreator.py` : calcule les **rendements implicites**, la **matrice de covariance**, l’**aversion au risque** et le paramètre **tau**.
- `BlackLittermanInputs.py` : génère les matrices **P** (opinions), **Q** (prévisions) et **Ω** (incertitudes) selon les méthodes **Idzorek** ou **Walter**.
- `BlackLittermanModel.py` : applique le modèle Black-Litterman pour produire les **rendements ajustés** et la **nouvelle matrice de covariance**.
- `PortfolioOptimization.py` : optimisations avec ou sans contraintes, en maximisant l’utilité ajustée du risque.
- `PlotFunctions.py` : propose des visualisations interactives (effet de tau, poids ajustés, surfaces 3D, etc.).
- `main.py` : script qui enchaîne les étapes : lecture des données, calculs, modèle BL, optimisation, visualisations.
---

**Objectif** : fournir une ressource pédagogique et pratique sur le modèle de Black-Litterman, utile à ceux qui sont intéressés par la gestion d’actifs.
