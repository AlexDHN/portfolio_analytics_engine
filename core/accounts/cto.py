from core.accounts.base import InvestmentAccount


class CTOAccount(InvestmentAccount):
    """
    Compte-Titres Ordinaire (CTO)
    """

    def __init__(self, name="CTO"):
        super().__init__(name)
