from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.optimize import linprog
from sklearn.covariance import LedoitWolf
from enum import Enum
import cvxpy as cp

class Weighting_Type(Enum):
    EQUAL_WEIGHT = "EW"

class BasePortfolio(ABC):
    """Classe de base pour les portefeuilles factoriels"""

    def __init__(self, df_factor: pd.DataFrame, 
                 target_factor: str, 
                 sensi_factors: list[str]):
        """
        Args:
            df_factor (pd.DataFrame): Données factorielles avec colonnes ['Date', 'Ticker', facteurs...]
            target_factor (str): Facteur à maximiser
            sensi_factors (list[str]): Liste des facteurs pour calculer les sensibilités
            weighting_type (WeightingType): Type de pondération
        """
        self.df_factor = df_factor
        self.target_factor = target_factor
        self.sensi_factors = sensi_factors

        # Vérifications
        required_cols = {'Date', 'Ticker'} | set(sensi_factors)
        if not required_cols.issubset(df_factor.columns):
            raise ValueError(f"Le DataFrame doit contenir les colonnes suivantes : {required_cols}")


    @abstractmethod
    def apply_weights(self, df_ptf: pd.DataFrame) -> pd.DataFrame:
        """Calcule le poids des actifs selon le portfolio type"""
        pass

    def construct_portfolio(self, df_factor: pd.DataFrame, rebalance_weight: bool = False, returns : pd.DataFrame  = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Construit un portefeuille fractile"""
        df_ptf = self.compute_zscore(df_factor)
        if rebalance_weight:
            df_ptf = self.apply_weights(df_ptf, returns)
        else:
            if 'Weight' not in df_ptf.columns:
                raise ValueError("Please Provide a Weight column")

        ptf_sensi = self.compute_factor_exposure(df_ptf)

        return df_ptf, ptf_sensi 

    def compute_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les z-scores"""
        for factor in self.sensi_factors:
            df[f'Zscore_{factor}'] = zscore(df[factor], nan_policy='omit').fillna(0)
        return df

    def compute_factor_exposure(self, df_ptf: pd.DataFrame) -> pd.Series:
        """Calcule l'exposition aux facteurs"""
        factor_exposures = {
            factor: (df_ptf['Weight'] * df_ptf[f'Zscore_{factor}']).sum()
            for factor in self.sensi_factors
        }
        return pd.Series(factor_exposures)


class FractilePortfolio(BasePortfolio):
    """Portefeuille basé sur une approche de classement en fractiles"""

    def __init__(self, df_factor: pd.DataFrame, 
                 target_factor: str, 
                 sensi_factors: list[str], 
                 nb_fractile: int = 4, 
                 weighting_type: Weighting_Type = Weighting_Type.EQUAL_WEIGHT):
        
        super().__init__(df_factor, target_factor, sensi_factors)
        self.nb_fractile = nb_fractile
        self.weighting_type = weighting_type
        if nb_fractile < 2:
            raise ValueError("Le nombre de fractiles doit être au moins 2.")


    def apply_weights(self, df_ptf: pd.DataFrame, returns : pd.DataFrame  = None) -> pd.DataFrame:
        """Applique la pondération long-short sur le portefeuille ainsi que le calcul des fractiles"""
        for factor in self.sensi_factors:
            df_ptf[f'Fractile_{factor}'] = pd.qcut(df_ptf[f'Zscore_{factor}'], q=self.nb_fractile, labels=range(1, self.nb_fractile + 1))
    
        num_long = len(df_ptf[df_ptf[f"Fractile_{self.target_factor}"] == self.nb_fractile])
        num_short = len(df_ptf[df_ptf[f"Fractile_{self.target_factor}"] == 1])

        if num_long == 0 or num_short == 0:
            raise ValueError("Un des fractiles est vide, impossible d'appliquer les poids.")

        df_ptf['Weight'] = df_ptf[f"Fractile_{self.target_factor}"].apply(
            lambda x: 1 / num_long if x == self.nb_fractile else -1 / num_short if x == 1 else 0
        )
        return df_ptf



class PureFactorPortfolio(BasePortfolio):
    """Portefeuille maximisant l'exposition à un facteur cible sous contraintes"""

    def apply_weights(self, df_ptf: pd.DataFrame, returns : pd.DataFrame  = None) -> pd.DataFrame:
        """Résout un problème d'optimisation linéaire pour obtenir les poids sous contraintes"""
        
        # Calcul de Σ (covariance shrinkée)
        lw = LedoitWolf()
        Sigma = lw.fit(returns.drop('Date', axis=1)).covariance_

        n_assets = len(df_ptf)
        factor_exposures = df_ptf[[f'Zscore_{factor}' for factor in self.sensi_factors]].values

        w = cp.Variable(n_assets)

        # Fonction objectif : Minimisation de la variance du portefeuille
        obj = cp.Minimize(cp.quad_form(w, Sigma))

        # Contraintes
        constraints = []

        # (1) Contraintes d'exposition factorielle : Target factor = 1, autres = 0
        target_idx = self.sensi_factors.index(self.target_factor)

        for i in range(len(self.sensi_factors)):  
            if i == target_idx:
                constraints.append(factor_exposures[:, i] @ w == 1)  # Exposition au facteur cible = 1
            else:
                constraints.append(factor_exposures[:, i] @ w == 0)  # Autres facteurs = 0

        # (2) Somme des poids = 0 (cash-neutral)
        constraints.append(cp.sum(w) == 0)

        # # (3) Neutralité sectorielle
        # sector_matrix = self.df_sectors.values  # (N stocks x S secteurs)
        # constraints.append(sector_matrix.T @ w == 0)

        # (4) Contraintes sur les poids (-1 ≤ w ≤ 1)
        constraints.append(w >= -1)
        constraints.append(w <= 1)

        # Résolution de l'optimisation quadratique
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP)

        # Résultats
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("L'optimisation n'a pas convergé")

        df_ptf['Weight'] = w.value
        return df_ptf