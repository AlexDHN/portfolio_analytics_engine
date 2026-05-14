import os
import time
import pandas as pd
import requests
import zipfile
import shutil
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from utils.logger import get_logger
from services.cache import load_cache, save_cache

logger = get_logger("SG_PEE")
CACHE_MAX_AGE = 86400  # 24h en secondes

CHROME_FOR_TESTING_JSON = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"


def get_chrome_version():
    """Retourne la version installée de Google Chrome"""
    try:
        output = subprocess.check_output(
            r'reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version',
            shell=True,
        ).decode("utf-8")
        version = output.strip().split()[-1]
        logger.info(f"Version Chrome détectée : {version}")
        return version
    except Exception as e:
        raise RuntimeError("Impossible de détecter la version de Chrome : " + str(e))


def get_driver_version(driver_path: str):
    """Retourne la version du ChromeDriver s’il existe, sinon None"""
    if not os.path.exists(driver_path):
        logger.info("ChromeDriver non trouvé")
        return None
    try:
        output = subprocess.check_output([driver_path, "--version"], shell=True).decode(
            "utf-8"
        )
        version = output.split()[1]
        logger.info(f"Version ChromeDriver détectée : {version}")
        return version
    except Exception:
        logger.warning("Impossible de détecter la version du ChromeDriver")
        return None


def update_chromedriver_if_needed(driver_path: str):
    """Met à jour ChromeDriver automatiquement si version incompatible"""
    driver_dir = os.path.dirname(driver_path)
    chrome_version = get_chrome_version()
    driver_version = get_driver_version(driver_path)

    if driver_version[:5] == chrome_version[:5]:
        logger.info(f"ChromeDriver déjà à jour ({driver_version})")
        return

    logger.info("Vérification / mise à jour de ChromeDriver...")
    r = requests.get(CHROME_FOR_TESTING_JSON)
    r.raise_for_status()
    data = r.json()

    stable_channel = data.get("channels", {}).get("Stable", {})
    downloads = stable_channel.get("downloads", {})
    chromedriver_downloads = downloads.get("chromedriver", [])

    win64_info = None
    if isinstance(chromedriver_downloads, list):
        for item in chromedriver_downloads:
            if item.get("platform") == "win64":
                win64_info = item
                break
    elif isinstance(chromedriver_downloads, dict):
        win64_info = chromedriver_downloads.get("win64")

    if not win64_info:
        raise RuntimeError(
            "Impossible de trouver l'URL du chromedriver-win64 dans le JSON"
        )

    download_url = win64_info.get("url")
    latest_version = win64_info.get("version") or win64_info.get("build_version")
    logger.info(f"Dernière version stable ChromeDriver : {latest_version}")
    logger.info(f"URL téléchargement : {download_url}")

    logger.info(f"Mise à jour de ChromeDriver vers {latest_version}...")
    zip_path = os.path.join(driver_dir, "chromedriver-win64.zip")
    r = requests.get(download_url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(
            f"Échec du téléchargement ({r.status_code}) : {download_url}"
        )

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    logger.info("Téléchargement terminé")

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        raise RuntimeError(
            f"Le fichier téléchargé n’est pas un zip valide : {download_url}"
        )

    # Extraction uniquement du fichier chromedriver.exe
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if filename == "chromedriver.exe":
                target_path = os.path.join(driver_dir, "chromedriver.exe")
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                logger.info(f"chromedriver.exe extrait vers {target_path}")
                break

    os.remove(zip_path)
    logger.info(f"ChromeDriver {latest_version} installé avec succès")


def get_sg_pee_data(
    isin: str, driver_path: str, headless: bool = True, cache_path: str = None
) -> pd.DataFrame:
    """
    Fetch fund data from SG29 Haussmann using Selenium + Highcharts,
    with optional caching to avoid repeated browser automation.

    Args:
        isin (str): The ISIN code of the fund (e.g., "QS0002904819").
        driver_path (str): Path to the ChromeDriver executable.
        headless (bool): Whether to run the browser in headless mode. Defaults to True.
        cache_path (str): Optional path to cache file.

    Returns:
        pd.DataFrame: DataFrame indexed by date with a 'value' column.
    """

    # 0. Chargement depuis cache si disponible
    if cache_path:
        cached_df = load_cache(cache_path, max_age_seconds=CACHE_MAX_AGE)
        if cached_df is not None:
            logger.info(f"Utilisation du cache PEE depuis {cache_path}")
            return cached_df

    # 1. Mise à jour automatique du ChromeDriver
    update_chromedriver_if_needed(driver_path)

    # 2. Récupération live
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
        logger.info("Mode headless activé")
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = f"https://sg29haussmann.societegenerale.fr/fr/nos-fonds/autres-fonds/details/isin/{isin}/"
    logger.info(f"Fetching PEE fund data from {url}")

    try:
        driver.get(url)
        driver.implicitly_wait(10)

        script = """
        var chart0 = Highcharts.charts[0];
        var chart1 = Highcharts.charts[1];
        var xData = chart0.series[0].xData;
        var yData = chart0.series[0].yData;
        var yData1 = chart1.series[0].yData;
        return [xData, yData, yData1];
        """
        x_data, y_data, y_data1 = driver.execute_script(script)

        if y_data and y_data[0] == 100:
            logger.info("Normalized data detected, using secondary Y axis (y_data1).")
            selected_data = y_data1
        else:
            selected_data = y_data

        df = pd.DataFrame(selected_data, index=x_data, columns=["value"])
        df.index = pd.to_datetime(df.index, unit="ms").normalize()
        logger.info(f"Retrieved {len(df)} data points for ISIN {isin}.")

        # 3. Sauvegarde dans le cache si chemin fourni
        if cache_path:
            save_cache(cache_path, df)

        return df

    except Exception as e:
        logger.error(f"Error fetching data for ISIN {isin}: {e}")
        raise

    finally:
        driver.quit()
