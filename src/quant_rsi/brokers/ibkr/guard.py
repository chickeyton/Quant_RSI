"""Paper-only guard. There is deliberately no option that disables it."""

from collections.abc import Iterable

from quant_rsi.brokers.ibkr.config import LIVE_PORTS, IbkrConfig

PAPER_PREFIX = "DU"


class LiveAccountRefused(RuntimeError):
    """Raised before any order intent is persisted or submitted."""


def check_paper(config: IbkrConfig, *, port: int, accounts: Iterable[str], client_id: int) -> str:
    """Return the verified paper account, or raise `LiveAccountRefused`."""
    accounts = tuple(accounts)
    reasons = []
    if config.account_mode != "paper":
        reasons.append("configuration is not paper mode")
    if port in LIVE_PORTS or port not in config.paper_ports:
        reasons.append(f"port {port} is not a registered paper port")
    if not accounts:
        reasons.append("TWS reported no managed accounts")
    elif not all(account.startswith(PAPER_PREFIX) for account in accounts):
        reasons.append("a managed account is not an individual paper (DU) account")
    if config.expected_account not in accounts:
        reasons.append("expected paper account is not managed by this session")
    if client_id != config.orders_client_id:
        reasons.append("session is not the registered order client")
    if reasons:
        raise LiveAccountRefused("live_account_refused: " + "; ".join(reasons))
    return config.expected_account
