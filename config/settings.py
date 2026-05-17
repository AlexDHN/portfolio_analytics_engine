import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

TODAY = datetime.today()
MIN_DATE = datetime(year=2023, month=5, day=1)

# === PARAMÈTRES GÉNÉRAUX ===
INVESTMENT_PATH = "investment.xlsx"
DRIVER_PATH = os.getenv("DRIVER_PATH")

# === CLÉS D'API ===
BDF_API_KEY = os.getenv("BDF_API_KEY")
BDF_HEADERS = {
    "Authorization": f"Apikey {BDF_API_KEY}",
    "accept": "application/json",
}

# === SOURCES DE DONNÉES ===
BDF_BASE_URL = "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/json"
KEY_INFLATION = "HICP.M.FR.N.000000.4D0.INX"  # "ICP.M.FR.N.000000.4.INX"  # "ICP.M.FR.N.000000.4.ANR"
KEY_LIVRET_A = "MIR1.M.FR.B.L23FRLA.D.R.A.2230U6.EUR.O"
ISIN_PEE = "QS0002904819"

# === PARAMÈTRES DE LOGIQUE ===
DEFAULT_LIVRET_TYPES = ["Livret A", "LDDS"]
FIFO_MATCHING = True  # Appliquer le matching FIFO pour ventes

# === LOGGING ===
LOG_LEVEL = "INFO"  # Niveau de log : DEBUG, INFO, WARNING, ERROR


@dataclass(frozen=True)
class InflationConfig:
    """
    Configuration immutable pour l'accès aux données d'inflation.
    """

    key: str
    base_url: str
    headers: dict
    start_date: pd.Timestamp
