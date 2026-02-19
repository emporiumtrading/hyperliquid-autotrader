"""REST client for Hyperliquid ``/info`` and ``/exchange`` endpoints.

Handles rate limiting, automatic retries with exponential backoff, EIP-712
signing for exchange actions, and structured logging via *structlog*.

Usage
-----
    from autotrader.hl.client import HLClient, create_client

    client = HLClient(account_address="0x...", api_wallet_key="0x...")
    meta = client.get_meta()
    book = client.get_l2_book("ETH")
"""

from __future__ import annotations

import time
from typing import Any

import requests
import structlog

from autotrader.hl.rate_limiter import acquire
from autotrader.hl.signing import sign_l1_action

logger = structlog.get_logger(__name__)

# Default retry configuration
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 0.5  # seconds
_BACKOFF_FACTOR = 2.0


class HLClient:
    """Synchronous REST client for the Hyperliquid API.

    Parameters
    ----------
    rest_url : str
        Base URL for the REST API.
    account_address : str
        The user's Ethereum-style address used as the default ``user``
        parameter in info queries.
    api_wallet_key : str
        Hex-encoded private key of the API wallet.  Used to sign
        ``/exchange`` payloads via EIP-712.
    vault_address : str
        Optional vault address.  If non-empty, exchange payloads will
        include ``vaultAddress``.
    """

    def __init__(
        self,
        rest_url: str = "https://api.hyperliquid.xyz",
        account_address: str = "",
        api_wallet_key: str = "",
        vault_address: str = "",
    ) -> None:
        self.rest_url = rest_url.rstrip("/")
        self.account_address = account_address
        self.api_wallet_key = api_wallet_key
        self.vault_address = vault_address
        self._is_mainnet: bool = "testnet" not in rest_url
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _post_info(self, payload: dict) -> Any:
        """POST to ``/info`` with rate limiting and retries.

        Parameters
        ----------
        payload : dict
            JSON body to send.

        Returns
        -------
        dict | list
            Decoded JSON response.

        Raises
        ------
        requests.HTTPError
            After exhausting all retry attempts.
        """
        url = f"{self.rest_url}/info"
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            acquire(weight=1.0)

            try:
                resp = self._session.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                last_exc = exc
                wait = _INITIAL_BACKOFF * (_BACKOFF_FACTOR**attempt)
                logger.warning(
                    "info_request_failed",
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    wait=wait,
                    error=str(exc),
                    payload_type=payload.get("type"),
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)

        # All retries exhausted
        logger.error(
            "info_request_exhausted",
            payload_type=payload.get("type"),
            error=str(last_exc),
        )
        raise last_exc  # type: ignore[misc]

    def _post_exchange(self, payload: dict) -> Any:
        """POST to ``/exchange`` with EIP-712 signature attached.

        The method takes a payload that *must* contain ``action`` and
        ``nonce`` keys.  It signs the action using the configured
        ``api_wallet_key`` and attaches ``signature`` (and optionally
        ``vaultAddress``) before sending.

        Parameters
        ----------
        payload : dict
            Must contain ``action`` (dict) and ``nonce`` (int).

        Returns
        -------
        dict
            Decoded JSON response.

        Raises
        ------
        ValueError
            If ``api_wallet_key`` is empty (cannot sign).
        """
        if not self.api_wallet_key:
            raise ValueError(
                "Cannot send exchange request: api_wallet_key is not configured"
            )

        action = payload.get("action", {})
        nonce = payload.get("nonce", 0)

        # Sign the action
        signature = sign_l1_action(
            wallet_key=self.api_wallet_key,
            action=action,
            nonce=nonce,
            vault_address=self.vault_address or None,
            is_mainnet=self._is_mainnet,
        )

        # Build the signed request body
        signed_payload: dict[str, Any] = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }
        if self.vault_address:
            signed_payload["vaultAddress"] = self.vault_address

        url = f"{self.rest_url}/exchange"
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            acquire(weight=1.0)

            try:
                resp = self._session.post(url, json=signed_payload, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                last_exc = exc
                wait = _INITIAL_BACKOFF * (_BACKOFF_FACTOR**attempt)
                logger.warning(
                    "exchange_request_failed",
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    wait=wait,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)

        logger.error(
            "exchange_request_exhausted",
            error=str(last_exc),
        )
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Info methods
    # ------------------------------------------------------------------

    def get_meta(self) -> dict:
        """Fetch exchange metadata (asset list, universe info)."""
        return self._post_info({"type": "meta"})

    def get_meta_and_asset_ctxs(self) -> list:
        """Fetch metadata together with per-asset context (funding, OI, etc.)."""
        return self._post_info({"type": "metaAndAssetCtxs"})

    def get_candle_snapshot(
        self,
        coin: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> list[dict]:
        """Fetch historical candles.

        Parameters
        ----------
        coin : str
            Asset symbol, e.g. ``"ETH"``.
        interval : str
            Candle interval, e.g. ``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``.
        start_time : int
            Start timestamp in milliseconds.
        end_time : int
            End timestamp in milliseconds.
        """
        return self._post_info(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time,
                },
            }
        )

    def get_l2_book(self, coin: str) -> dict:
        """Fetch the current L2 order book for *coin*."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def get_user_state(self, address: str | None = None) -> dict:
        """Fetch clearinghouse state (positions, margin, etc.) for a user."""
        return self._post_info(
            {
                "type": "clearinghouseState",
                "user": address or self.account_address,
            }
        )

    def get_user_fills(self, address: str | None = None) -> list:
        """Fetch recent fills for a user."""
        return self._post_info(
            {
                "type": "userFills",
                "user": address or self.account_address,
            }
        )

    def get_open_orders(self, address: str | None = None) -> list:
        """Fetch open orders for a user."""
        return self._post_info(
            {
                "type": "openOrders",
                "user": address or self.account_address,
            }
        )

    def get_order_status(self, oid: int) -> dict:
        """Fetch the status of a specific order by its ID."""
        return self._post_info(
            {
                "type": "orderStatus",
                "user": self.account_address,
                "oid": oid,
            }
        )

    def get_funding_history(
        self,
        coin: str,
        start_time: int,
        end_time: int | None = None,
    ) -> list:
        """Fetch funding rate history for *coin*.

        Parameters
        ----------
        coin : str
            Asset symbol.
        start_time : int
            Start timestamp in milliseconds.
        end_time : int | None
            End timestamp in milliseconds.  Defaults to *now*.
        """
        return self._post_info(
            {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": start_time,
                "endTime": end_time or int(time.time() * 1000),
            }
        )

    # ------------------------------------------------------------------
    # Exchange methods
    # ------------------------------------------------------------------

    def place_order(self, order: dict) -> dict:
        """Place an order on the exchange.

        Parameters
        ----------
        order : dict
            Full order payload including ``action``, ``nonce``, and
            ``signature`` fields.
        """
        logger.info("place_order", order=order)
        return self._post_exchange(order)

    def cancel_order(self, asset: int, oid: int) -> dict:
        """Cancel a single order.

        Parameters
        ----------
        asset : int
            Asset index.
        oid : int
            Order ID to cancel.
        """
        payload = {
            "action": {
                "type": "cancel",
                "cancels": [{"asset": asset, "oid": oid}],
            },
        }
        logger.info("cancel_order", asset=asset, oid=oid)
        return self._post_exchange(payload)

    def cancel_all(self) -> dict:
        """Cancel all open orders for the configured account."""
        payload = {
            "action": {
                "type": "cancelByCloid",
            },
        }
        logger.info("cancel_all")
        return self._post_exchange(payload)


# ----------------------------------------------------------------------
# Module-level factory
# ----------------------------------------------------------------------


def create_client(config: dict) -> HLClient:
    """Create an :class:`HLClient` from a configuration dictionary.

    Expected keys (all optional, sensible defaults are used):

    * ``rest_url`` -- Base REST URL.
    * ``account_address`` -- User Ethereum address.
    * ``api_wallet_key`` -- API wallet private key.
    * ``vault_address`` -- Optional vault address for vault-mode trading.
    """
    return HLClient(
        rest_url=config.get("rest_url", "https://api.hyperliquid.xyz"),
        account_address=config.get("account_address", ""),
        api_wallet_key=config.get("api_wallet_key", ""),
        vault_address=config.get("vault_address", ""),
    )
