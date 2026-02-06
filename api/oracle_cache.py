# api/oracle_cache.py
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


@dataclass
class CacheHit:
    row: Dict[str, Any]
    expires_at_utc: int
    fetched_at_utc: int


class OracleCache:
    """
    SQLite cache for Oracle-computed features.

    Schema goal:
      - easy human inspection (market/ticker columns)
      - safe TTL invalidation (expires_at_utc)
      - robust to older schemas (auto-migrate by recreating table)
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @staticmethod
    def _now() -> int:
        return int(time.time())

    @staticmethod
    def make_key(
        asset_type: str,
        market: str,
        ticker: str,
        lookback_days: int,
        source: str = "yfinance",
        version: str = "v1",
    ) -> str:
        a = (asset_type or "").strip().lower()
        m = (market or "").strip().upper()
        t = (ticker or "").strip().upper()
        return f"{version}|{source}|{a}|{m}|{t}|lb{int(lookback_days)}"

    def _ensure_schema(self) -> None:
        """
        Create table if missing. If table exists but missing expected columns,
        migrate by renaming the old table and creating a fresh one.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='oracle_cache';"
        )
        exists = cur.fetchone() is not None

        expected_cols = {
            "cache_key",
            "asset_type",
            "market",
            "ticker",
            "lookback_days",
            "source",
            "fetched_at_utc",
            "expires_at_utc",
            "row_json",
        }

        if exists:
            cur.execute("PRAGMA table_info(oracle_cache);")
            cols = {r[1] for r in cur.fetchall()}  # r[1] = name
            if not expected_cols.issubset(cols):
                # migrate: rename old table
                ts = self._now()
                cur.execute(f"ALTER TABLE oracle_cache RENAME TO oracle_cache_old_{ts};")
                self._conn.commit()

        # create fresh schema
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS oracle_cache (
              cache_key      TEXT PRIMARY KEY,
              asset_type     TEXT NOT NULL,
              market         TEXT NOT NULL,
              ticker         TEXT NOT NULL,
              lookback_days  INTEGER NOT NULL,
              source         TEXT NOT NULL,
              fetched_at_utc INTEGER NOT NULL,
              expires_at_utc INTEGER NOT NULL,
              row_json       TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_oracle_cache_lookup ON oracle_cache (market, ticker);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_oracle_cache_expiry ON oracle_cache (expires_at_utc);")
        self._conn.commit()

    def get(
        self,
        asset_type: str,
        market: str,
        ticker: str,
        lookback_days: int,
        source: str = "yfinance",
    ) -> Optional[CacheHit]:
        key = self.make_key(asset_type, market, ticker, lookback_days, source=source)
        now = self._now()

        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT row_json, expires_at_utc, fetched_at_utc
            FROM oracle_cache
            WHERE cache_key = ?
            """,
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None

        row_json, expires_at, fetched_at = row
        try:
            expires_at = int(expires_at)
            fetched_at = int(fetched_at)
        except Exception:
            return None

        if expires_at <= now:
            # expired -> delete
            try:
                cur.execute("DELETE FROM oracle_cache WHERE cache_key = ?", (key,))
                self._conn.commit()
            except Exception:
                pass
            return None

        try:
            payload = json.loads(row_json)
            if not isinstance(payload, dict):
                return None
        except Exception:
            return None

        return CacheHit(row=payload, expires_at_utc=expires_at, fetched_at_utc=fetched_at)

    def set(
        self,
        asset_type: str,
        market: str,
        ticker: str,
        lookback_days: int,
        row: Dict[str, Any],
        expires_at_utc: int,
        source: str = "yfinance",
    ) -> None:
        key = self.make_key(asset_type, market, ticker, lookback_days, source=source)
        now = self._now()
        payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO oracle_cache (cache_key, asset_type, market, ticker, lookback_days, source, fetched_at_utc, expires_at_utc, row_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
              fetched_at_utc = excluded.fetched_at_utc,
              expires_at_utc = excluded.expires_at_utc,
              row_json = excluded.row_json
            """,
            (
                key,
                (asset_type or "").strip().lower(),
                (market or "").strip().upper(),
                (ticker or "").strip().upper(),
                int(lookback_days),
                str(source),
                int(now),
                int(expires_at_utc),
                payload,
            ),
        )
        self._conn.commit()

    def table_columns(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("PRAGMA table_info(oracle_cache);")
        return [r[1] for r in cur.fetchall()]

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT market, ticker, fetched_at_utc, expires_at_utc, lookback_days, source
            FROM oracle_cache
            ORDER BY fetched_at_utc DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        out = []
        for m, t, fa, ea, lb, src in cur.fetchall():
            out.append(
                {
                    "market": m,
                    "ticker": t,
                    "fetched_at_utc": int(fa),
                    "expires_at_utc": int(ea),
                    "lookback_days": int(lb),
                    "source": src,
                }
            )
        return out

