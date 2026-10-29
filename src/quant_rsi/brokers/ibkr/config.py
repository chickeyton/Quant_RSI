"""Local IBKR connection settings; stored outside version control in `.quant_rsi/ibkr.json`."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, model_validator

from quant_rsi.core.contracts_base import StrictModel

LIVE_PORTS = frozenset({7496, 4001})
DEFAULT_PAPER_PORTS = (7497, 4002)


class IbkrConfig(StrictModel):
    schema_version: Literal[1]
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(gt=0, lt=65536)
    data_client_id: int = Field(ge=1, le=32767)
    orders_client_id: int = Field(ge=1, le=32767)
    account_mode: Literal["paper"]
    expected_account: str = Field(pattern=r"^DU[0-9]{1,16}$")
    paper_ports: tuple[int, ...] = DEFAULT_PAPER_PORTS
    max_clock_skew_seconds: float = Field(default=2.0, gt=0, le=60)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    restart_time: str = Field(default="23:45", pattern=r"^[0-2][0-9]:[0-5][0-9]$")
    restart_time_zone: str = "America/New_York"
    restart_guard_seconds: int = Field(default=900, ge=0, le=7200)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.data_client_id == self.orders_client_id:
            raise ValueError("data and order roles need separate client IDs")
        if not self.paper_ports or set(self.paper_ports) & LIVE_PORTS:
            raise ValueError("paper ports must be nonempty and exclude the default live ports")
        if int(self.restart_time[:2]) > 23:
            raise ValueError("restart_time must be HH:MM")
        try:
            ZoneInfo(self.restart_time_zone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("unknown restart time zone") from error
        return self


def in_restart_window(config: IbkrConfig, now: datetime) -> bool:
    """True within `restart_guard_seconds` of the configured daily TWS/Gateway restart."""
    zone = ZoneInfo(config.restart_time_zone)
    local = now.astimezone(zone)
    hour, minute = (int(x) for x in config.restart_time.split(":"))
    guard = timedelta(seconds=config.restart_guard_seconds)
    for offset in (-1, 0, 1):
        day = local.date() + timedelta(days=offset)
        restart = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
        if abs(local - restart) <= guard:
            return True
    return False


def load_config(path: Path) -> IbkrConfig:
    return IbkrConfig.model_validate_json(Path(path).read_bytes())
