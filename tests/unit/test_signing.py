"""Tests for EIP-712 signing module."""

from autotrader.hl.signing import (
    _action_hash,
    _build_domain_separator,
    sign_l1_action,
)

# Deterministic test key (never use in production)
_TEST_KEY = "0x" + "ab" * 32


def test_sign_l1_action_returns_r_s_v() -> None:
    action = {"type": "order", "orders": [], "grouping": "na"}
    sig = sign_l1_action(_TEST_KEY, action, nonce=1, is_mainnet=True)

    assert "r" in sig
    assert "s" in sig
    assert "v" in sig
    assert sig["r"].startswith("0x")
    assert sig["s"].startswith("0x")
    assert sig["v"] in (27, 28)


def test_sign_l1_action_deterministic() -> None:
    action = {"type": "cancel", "cancels": [{"asset": 0, "oid": 123}]}
    sig1 = sign_l1_action(_TEST_KEY, action, nonce=42, is_mainnet=True)
    sig2 = sign_l1_action(_TEST_KEY, action, nonce=42, is_mainnet=True)
    assert sig1 == sig2


def test_different_nonces_produce_different_signatures() -> None:
    action = {"type": "order", "orders": [], "grouping": "na"}
    sig1 = sign_l1_action(_TEST_KEY, action, nonce=1, is_mainnet=True)
    sig2 = sign_l1_action(_TEST_KEY, action, nonce=2, is_mainnet=True)
    assert sig1 != sig2


def test_mainnet_vs_testnet_different_signatures() -> None:
    action = {"type": "order", "orders": [], "grouping": "na"}
    sig_main = sign_l1_action(_TEST_KEY, action, nonce=1, is_mainnet=True)
    sig_test = sign_l1_action(_TEST_KEY, action, nonce=1, is_mainnet=False)
    assert sig_main != sig_test


def test_action_hash_changes_with_vault() -> None:
    action = {"type": "order", "orders": [], "grouping": "na"}
    h1 = _action_hash(action, None, 1)
    h2 = _action_hash(action, "0x" + "cc" * 20, 1)
    assert h1 != h2


def test_domain_separator_different_chains() -> None:
    ds1 = _build_domain_separator(1337)
    ds2 = _build_domain_separator(13337)
    assert ds1 != ds2
    assert len(ds1) == 32
    assert len(ds2) == 32


def test_sign_with_vault_address() -> None:
    action = {"type": "order", "orders": [], "grouping": "na"}
    vault = "0x" + "dd" * 20
    sig = sign_l1_action(
        _TEST_KEY, action, nonce=1, vault_address=vault, is_mainnet=True
    )
    assert "r" in sig and "s" in sig and "v" in sig


def test_sign_key_without_0x_prefix() -> None:
    key_no_prefix = "ab" * 32
    action = {"type": "order", "orders": [], "grouping": "na"}
    sig = sign_l1_action(key_no_prefix, action, nonce=1, is_mainnet=True)
    sig2 = sign_l1_action("0x" + key_no_prefix, action, nonce=1, is_mainnet=True)
    assert sig == sig2
