from core.accounts.base import InvestmentAccount


class PEEAccount(InvestmentAccount):
    """
    Plan d'Épargne Entreprise (PEE)
    """

    def __init__(self, name="PEE"):
        super().__init__(name)
