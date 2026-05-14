from typing import List, Dict
import pandas as pd
from core.accounts import InvestmentAccount


class Portfolio:
    """
    Represents a complete investment portfolio, aggregating multiple accounts (supports).
    """

    def __init__(self, name: str):
        self.name = name
        self.accounts: List[InvestmentAccount] = []

    def add_account(self, account: InvestmentAccount):
        """
        Add an investment account (support) to the portfolio.
        """
        self.accounts.append(account)

    def get_account_by_name(self, name: str) -> InvestmentAccount:
        """
        Retrieve a specific account by its name.
        """
        for acc in self.accounts:
            if acc.name == name:
                return acc
        raise ValueError(f"Account '{name}' not found in portfolio.")

    def total_value(self, date: pd.Timestamp) -> float:
        """
        Compute the total value of the portfolio at a given date.
        """
        return sum(account.total_value(date) for account in self.accounts)

    def account_values(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Return the value of each account in the portfolio at the given date.
        """
        return {account.name: account.total_value(date) for account in self.accounts}

    def all_returns(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute returns of each investment in every account.
        Returns a nested dict: {account_name: {investment_name: returns_df}}
        """
        portfolio_returns = {}
        for account in self.accounts:
            returns = account.all_returns()
            portfolio_returns[account.name] = returns
        return portfolio_returns

    def summary(self, date: pd.Timestamp):
        """
        Print a high-level summary of the portfolio at a given date.
        """
        print(f"📊 Portfolio Summary: {self.name} @ {date.date()}")
        print("-" * 40)
        for account in self.accounts:
            value = account.total_value(date)
            print(f"{account.name:<20}: {value:,.2f} €")
        print("-" * 40)
        print(f"{'Total':<20}: {self.total_value(date):,.2f} €")
