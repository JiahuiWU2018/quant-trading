"""Fama-French factor return data connector.

Fetches factor return data directly from the Kenneth French Data Library
(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
by downloading the public ZIP/CSV files over HTTPS.  No third-party data
reader package is required — only the Python standard library plus ``requests``
(already a transitive dependency of yfinance).

Factor returns are stored as percentages in the source files and are
converted to decimals (divide by 100) before returning.

Supported factor families are defined in optim.factor.FACTOR_FAMILIES.
The Carhart 4-factor set is assembled by joining FF3 + MOM datasets.

Caching
-------
Downloads are cached in-process (per connector instance).  The full dataset
is downloaded once and then date-filtered locally on each call, so subsequent
calls with different date ranges are instant.

Public/private boundary
-----------------------
This connector lives in the public repo because it only fetches freely
available academic data.  The private repo is responsible for:
  - Deciding which factor family to use
  - Running compute_factor_loadings() on private strategy returns
  - Storing the resulting FactorModel
"""

import io
import logging
import zipfile
from datetime import date, datetime
from typing import Union

import pandas as pd
import requests

from quant_trading.optim.factor import FACTOR_FAMILIES, FACTOR_NAMES

logger = logging.getLogger(__name__)

_FF_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

_FF_ZIP_MAP: dict[str, str] = {
    "FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "MOM": "F-F_Momentum_Factor_daily_CSV.zip",
}

_FF3_COLS = ["Mkt-RF", "SMB", "HML"]
_FF5_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

DateLike = Union[str, date, datetime]
_TIMEOUT = 30


class FamaFrenchConnector:
    """Download and cache Fama-French factor returns.

    Downloads ZIP files directly from the Kenneth French Data Library over
    HTTPS.  No API key is required.  The full dataset is fetched once per
    family per connector lifetime; subsequent calls with different date ranges
    are filtered locally.

    Args:
        timeout: HTTP request timeout in seconds.

    Example:
        connector = FamaFrenchConnector()
        ff3 = connector.fetch_factor_returns(
            family=FACTOR_FAMILIES.FF3,
            start="2020-01-01",
        )
        print(ff3.head())
        #             Mkt-RF     SMB     HML
        # 2020-01-02  -0.0007 -0.0030  0.0040
    """

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self._timeout = timeout
        self._full_cache: dict[str, pd.DataFrame] = {}

    def fetch_factor_returns(
        self,
        family: FACTOR_FAMILIES,
        start: DateLike = "2000-01-01",
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Download daily Fama-French factor returns as decimals.

        Args:
            family: Which factor family to fetch.  CUSTOM is not supported.
            start: Start date (inclusive).
            end: End date (inclusive).  Defaults to today if None.

        Returns:
            DataFrame with DatetimeIndex and factor return columns as decimals.

        Raises:
            ValueError: If family is CUSTOM.
            RuntimeError: If the HTTP download fails.
        """
        if family == FACTOR_FAMILIES.CUSTOM:
            raise ValueError(
                "FACTOR_FAMILIES.CUSTOM cannot be fetched by FamaFrenchConnector. "
                "The private repository must supply its own factor returns DataFrame."
            )

        end_date = str(end) if end else date.today().isoformat()
        df = self._get_full(family)
        filtered = df.loc[str(start): end_date]

        logger.info(
            "FamaFrenchConnector: %s, %d rows (%s to %s)",
            family.value,
            len(filtered),
            filtered.index.min().date() if len(filtered) else "N/A",
            filtered.index.max().date() if len(filtered) else "N/A",
        )
        return filtered

    def fetch_risk_free_rate(
        self,
        start: DateLike = "2000-01-01",
        end: DateLike | None = None,
    ) -> pd.Series:
        """Fetch the daily risk-free rate (RF) from the FF3 dataset as decimals.

        Args:
            start: Start date.
            end: End date.

        Returns:
            Series with DatetimeIndex and daily RF values as decimals.
        """
        end_date = str(end) if end else date.today().isoformat()
        raw = self._download_csv("FF3")
        rf = raw["RF"] / 100.0
        rf.name = "RF"
        return rf.loc[str(start): end_date]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_full(self, family: FACTOR_FAMILIES) -> pd.DataFrame:
        """Return the full unfiltered dataset for a family, using in-process cache."""
        key = family.value
        if key in self._full_cache:
            return self._full_cache[key]

        if family == FACTOR_FAMILIES.CARHART4:
            ff3_raw = self._download_csv("FF3")
            mom_raw = self._download_csv("MOM")
            df = ff3_raw[_FF3_COLS].join(mom_raw, how="inner")
            mom_col = [c for c in df.columns if c not in _FF3_COLS]
            if mom_col:
                df = df.rename(columns={mom_col[0]: "MOM"})
            df = df / 100.0
            canonical = FACTOR_NAMES.get(family, list(df.columns))
            df.columns = canonical[: len(df.columns)]
        elif family == FACTOR_FAMILIES.FF5:
            raw = self._download_csv("FF5")
            df = raw[_FF5_COLS] / 100.0
            df.columns = FACTOR_NAMES[family]
        else:
            raw = self._download_csv("FF3")
            df = raw[_FF3_COLS] / 100.0
            df.columns = FACTOR_NAMES[family]

        self._full_cache[key] = df
        return df

    def _download_csv(self, dataset_key: str) -> pd.DataFrame:
        """Download and parse one French Data Library ZIP/CSV file.

        Args:
            dataset_key: Key into _FF_ZIP_MAP ("FF3", "FF5", or "MOM").

        Returns:
            Raw DataFrame with DatetimeIndex and original column names
            (values still in percentage units).

        Raises:
            RuntimeError: If the HTTP request fails or parsing fails.
        """
        zip_name = _FF_ZIP_MAP[dataset_key]
        url = f"{_FF_BASE_URL}/{zip_name}"
        logger.debug("FamaFrenchConnector: downloading %s", url)

        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download Fama-French data from {url}: {exc}"
            ) from exc

        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_name = next(
                n for n in zf.namelist() if n.upper().endswith(".CSV")
            )
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract CSV from {zip_name}: {exc}"
            ) from exc

        return self._parse_ff_csv(raw_text)

    @staticmethod
    def _parse_ff_csv(text: str) -> pd.DataFrame:
        """Parse the French Data Library file format.

        Format (confirmed from live downloads):
          - Description lines at the top (plain text, no commas on most lines)
          - Column header row: ``,Mkt-RF,SMB,HML,RF``  (leading comma, empty date field)
          - Daily data rows: ``19260701,    0.09,   -0.25, ...``  (comma-delimited)
          - Annual summary rows at the end: ``1926,  0.03, ...``  (4-digit year, skipped)

        Args:
            text: Raw file content from the ZIP archive.

        Returns:
            DataFrame with DatetimeIndex and float columns (percentage units).
        """
        lines = text.splitlines()
        header: list[str] | None = None
        data_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            parts = [p.strip() for p in stripped.split(",")]
            first_tok = parts[0]

            if header is None:
                # Header row: first token is empty (leading comma) and has ≥2 parts
                if first_tok == "" and len(parts) >= 2:
                    header = [p for p in parts[1:] if p]  # skip the empty first field
                continue

            # 8-digit date rows are daily data; skip 4-digit annual summary rows
            if len(first_tok) == 8 and first_tok.isdigit():
                data_lines.append(stripped)

        if header is None or not data_lines:
            raise RuntimeError(
                "Could not parse Fama-French file: no data rows found."
            )

        rows = [[p.strip() for p in line.split(",")] for line in data_lines]
        # Each row: [date, val1, val2, ...]
        df = pd.DataFrame(rows, columns=["Date"] + header[: len(rows[0]) - 1])
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df.set_index("Date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="all")
        return df
