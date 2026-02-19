"""EIP-712 signing for Hyperliquid exchange actions.

Implements the Hyperliquid-specific EIP-712 signing scheme required for
all ``/exchange`` API calls.  The approach follows the Hyperliquid Python
SDK conventions:

1. Construct a *connection id* from the action + nonce + vault address.
2. Build an EIP-712 typed-data structure with the correct domain separator
   (chain id 1337 for mainnet, 13337 for testnet).
3. Hash the structured data and sign with the API wallet private key.
4. Return the ``{"r": ..., "s": ..., "v": ...}`` signature dict that the
   exchange expects.

References
----------
* Hyperliquid docs on signing: https://hyperliquid.gitbook.io/hyperliquid-docs
* The official Python SDK uses ``eth_account`` + ``encode_structured_data``.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from eth_abi import encode as abi_encode
from eth_account import Account
from eth_utils import keccak

logger = structlog.get_logger(__name__)

# Hyperliquid chain IDs
_MAINNET_CHAIN_ID = 1337
_TESTNET_CHAIN_ID = 13337

# EIP-712 domain name and version used by Hyperliquid
_DOMAIN_NAME = "Exchange"
_DOMAIN_VERSION = "1"
_VERIFYING_CONTRACT = "0x0000000000000000000000000000000000000000"

# Pre-computed type hashes (keccak256 of the type string)
_EIP712_DOMAIN_TYPEHASH = keccak(
    text="EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
)

# Hyperliquid "Agent" type for L1 actions
_AGENT_TYPEHASH = keccak(
    text="Agent(address source,bytes32 connectionId)"
)


def _action_hash(action: dict, vault_address: str | None, nonce: int) -> bytes:
    """Compute the connection id / action hash.

    Hyperliquid hashes the JSON-encoded action concatenated with the nonce
    and vault address to produce a 32-byte connection id.

    Parameters
    ----------
    action : dict
        The action payload (e.g. order, cancel).
    vault_address : str | None
        Vault address if operating in vault mode, else ``None``.
    nonce : int
        Monotonically increasing nonce.

    Returns
    -------
    bytes
        32-byte keccak256 hash used as ``connectionId``.
    """
    # Canonical JSON encoding (sorted keys, no whitespace) following the SDK
    action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)

    # The connection id is keccak(action_json + nonce + vault_address_flag)
    data = action_json.encode("utf-8")
    data += nonce.to_bytes(8, "big")
    if vault_address:
        data += b"\x01"
        # Normalize vault address to bytes20
        vault_bytes = bytes.fromhex(vault_address.replace("0x", ""))
        data += vault_bytes
    else:
        data += b"\x00"

    return keccak(data)


def _build_domain_separator(chain_id: int) -> bytes:
    """Build the EIP-712 domain separator hash.

    Parameters
    ----------
    chain_id : int
        1337 for mainnet, 13337 for testnet.

    Returns
    -------
    bytes
        32-byte domain separator hash.
    """
    return keccak(
        _EIP712_DOMAIN_TYPEHASH
        + keccak(text=_DOMAIN_NAME)
        + keccak(text=_DOMAIN_VERSION)
        + abi_encode(["uint256"], [chain_id])
        + abi_encode(["address"], [_VERIFYING_CONTRACT])
    )


def _build_struct_hash(source: str, connection_id: bytes) -> bytes:
    """Build the EIP-712 struct hash for the Agent type.

    Parameters
    ----------
    source : str
        The signer's address (the API wallet address, derived from key).
    connection_id : bytes
        32-byte connection id from :func:`_action_hash`.

    Returns
    -------
    bytes
        32-byte struct hash.
    """
    return keccak(
        _AGENT_TYPEHASH
        + abi_encode(["address"], [source])
        + connection_id  # already 32 bytes, used directly
    )


def sign_l1_action(
    wallet_key: str,
    action: dict,
    nonce: int,
    vault_address: str | None = None,
    is_mainnet: bool = True,
) -> dict[str, Any]:
    """Sign a Hyperliquid L1 action using EIP-712.

    Parameters
    ----------
    wallet_key : str
        Hex-encoded private key (with or without ``0x`` prefix).
    action : dict
        The action payload.
    nonce : int
        Request nonce.
    vault_address : str | None
        Optional vault address.
    is_mainnet : bool
        ``True`` for mainnet (chain 1337), ``False`` for testnet (13337).

    Returns
    -------
    dict
        Signature dict with keys ``r``, ``s``, ``v`` as hex strings /
        integer suitable for the exchange API.
    """
    # Normalize key
    if not wallet_key.startswith("0x"):
        wallet_key = "0x" + wallet_key

    # Derive signer address
    account = Account.from_key(wallet_key)
    source = account.address

    # Compute connection id
    connection_id = _action_hash(action, vault_address, nonce)

    # Build EIP-712 digest
    chain_id = _MAINNET_CHAIN_ID if is_mainnet else _TESTNET_CHAIN_ID
    domain_separator = _build_domain_separator(chain_id)
    struct_hash = _build_struct_hash(source, connection_id)

    # EIP-712: "\x19\x01" + domainSeparator + structHash
    digest = keccak(b"\x19\x01" + domain_separator + struct_hash)

    # Sign the digest
    signed = account.unsafe_sign_hash(digest)

    signature = {
        "r": hex(signed.r),
        "s": hex(signed.s),
        "v": signed.v,
    }

    logger.debug(
        "signed_l1_action",
        action_type=action.get("type"),
        nonce=nonce,
        signer=source,
    )

    return signature
