"""Pydantic-модели пайплайна.

Здесь же живут модели конфига: конфиг читается один раз при старте и
дальше ходит по пайплайну типизированным объектом, а не словарём.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Токен и метрики
# --------------------------------------------------------------------------


class Token(BaseModel):
    """Новый токен на бондинговой кривой pump.fun."""

    model_config = ConfigDict(extra="allow")

    mint: str
    name: str | None = None
    symbol: str | None = None
    description: str | None = None
    image_uri: str | None = None
    metadata_uri: str | None = None
    twitter: str | None = None
    telegram: str | None = None
    website: str | None = None

    creator: str | None = None
    created_timestamp: float = 0.0

    unique_buyers: int = 0
    curve_progress: float = 0.0          # 0..1, доля выкупленной кривой
    market_cap_sol: float = 0.0
    sol_in_curve: float = 0.0

    @property
    def age_seconds(self) -> float:
        if not self.created_timestamp:
            return 0.0
        return max(0.0, time.time() - self.created_timestamp)

    @property
    def has_metadata(self) -> bool:
        return bool(self.name) and bool(self.image_uri)

    @property
    def has_socials(self) -> bool:
        return any([self.twitter, self.telegram, self.website])


class Holder(BaseModel):
    """Держатель токена."""

    model_config = ConfigDict(extra="allow")

    address: str
    amount: float = 0.0
    share: float = 0.0                   # доля от общего предложения, 0..1
    is_creator: bool = False


class Trade(BaseModel):
    """Сделка на кривой."""

    model_config = ConfigDict(extra="allow")

    signature: str | None = None
    wallet: str
    is_buy: bool = True
    sol_amount: float = 0.0
    token_amount: float = 0.0
    timestamp: float = 0.0
    slot: int | None = None


class TokenMetrics(BaseModel):
    """Метрики, посчитанные кодом в analyzer.py. Без LLM."""

    top5_share: float = 0.0              # доля топ-5 кошельков, 0..1
    creator_share: float = 0.0           # доля создателя, 0..1
    sniper_count: int = 0                # покупки в первые секунды жизни
    wallet_diversity: float = 0.0        # 0..1, чем выше, тем разнообразнее
    social_signals: float = 0.0          # 0..1, наличие и качество ссылок
    curve_health: float = 0.0            # 0..1, ровность набора кривой
    buy_sell_ratio: float = 0.0
    unique_wallets: int = 0
    trade_count: int = 0
    risk_score: float = 10.0             # 0..10, чем выше, тем хуже

    @property
    def quality(self) -> float:
        """Сводное качество метрик 0..1 — компонент `metrics` в скоринге."""
        return max(0.0, min(1.0, 1.0 - self.risk_score / 10.0))


# --------------------------------------------------------------------------
# Ответы агентов
# --------------------------------------------------------------------------


class AuditResult(BaseModel):
    """Агент-аудитор: паттерны, которые не видно в агрегированных метриках."""

    coordinated_buying: bool = True
    wash_trading: bool = True
    creator_dump_prep: bool = True
    bundled_launch: bool = True
    organic_buyer_share: float = 0.0     # 0..1
    confidence: float = 0.0              # 0..1
    flags: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @property
    def score(self) -> float:
        """0..1: органика минус штраф за каждый сработавший флаг."""
        penalties = sum(
            0.25
            for flag in (
                self.coordinated_buying,
                self.wash_trading,
                self.creator_dump_prep,
                self.bundled_launch,
            )
            if flag
        )
        return max(0.0, min(1.0, self.organic_buyer_share - penalties))

    @classmethod
    def pessimistic(cls, reason: str) -> AuditResult:
        """Фолбэк при ошибке: всё плохо, органики нет."""
        return cls(
            coordinated_buying=True,
            wash_trading=True,
            creator_dump_prep=True,
            bundled_launch=True,
            organic_buyer_share=0.0,
            confidence=0.0,
            flags=["agent_failure"],
            reasoning=reason,
        )


class NarrativeResult(BaseModel):
    """Агент-нарратив: мем-потенциал."""

    trend_fit: float = 0.0               # попадание в тренд, 0..1
    virality: float = 0.0                # виральность, 0..1
    community_signals: float = 0.0       # признаки живого сообщества, 0..1
    launch_timing: float = 0.0           # своевременность запуска, 0..1
    reasoning: str = ""

    @property
    def score(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                (self.trend_fit + self.virality + self.community_signals + self.launch_timing)
                / 4.0,
            ),
        )

    @classmethod
    def pessimistic(cls, reason: str) -> NarrativeResult:
        return cls(reasoning=reason)


class TimingResult(BaseModel):
    """Агент-тайминг: состояние рынка, а не конкретный токен."""

    market_sentiment: float = 0.0        # 0..1
    meme_season: float = 0.0             # 0..1
    volume_level: float = 0.0            # 0..1
    anomalies: list[str] = Field(default_factory=list)
    reasoning: str = ""
    fetched_at: float = 0.0

    @property
    def score(self) -> float:
        base = (self.market_sentiment + self.meme_season + self.volume_level) / 3.0
        penalty = 0.1 * len(self.anomalies)
        return max(0.0, min(1.0, base - penalty))

    @classmethod
    def pessimistic(cls, reason: str) -> TimingResult:
        return cls(anomalies=["agent_failure"], reasoning=reason)


class CheckerResult(BaseModel):
    """Адверсариальный чекер: ищет причины НЕ покупать."""

    approve: bool = False
    reason: str = ""
    flags: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def pessimistic(cls, reason: str) -> CheckerResult:
        """Ошибка проверки равна отказу, а не молчаливому пропуску."""
        return cls(approve=False, reason=reason, flags=["agent_failure"], confidence=0.0)


# --------------------------------------------------------------------------
# Скоринг, решение, позиция
# --------------------------------------------------------------------------


class Scores(BaseModel):
    """Разложенный скоринг: компоненты и итог."""

    audit: float = 0.0
    narrative: float = 0.0
    timing: float = 0.0
    metrics: float = 0.0
    total: float = 0.0


class Analysis(BaseModel):
    """Всё, что пайплайн узнал о токене к моменту решения."""

    token: Token
    metrics: TokenMetrics = Field(default_factory=TokenMetrics)
    audit: AuditResult | None = None
    narrative: NarrativeResult | None = None
    timing: TimingResult | None = None
    scores: Scores = Field(default_factory=Scores)
    checker: CheckerResult | None = None


class Position(BaseModel):
    """Открытая позиция."""

    mint: str
    symbol: str | None = None
    entry_price: float = 0.0
    sol_spent: float = 0.0
    token_amount: float = 0.0
    opened_at: float = 0.0
    tx_hash: str = ""
    score: float = 0.0


class TradeDecision(BaseModel):
    """Решение риск-гейта."""

    approved: bool
    size_sol: float = 0.0
    reason: str = ""


# --------------------------------------------------------------------------
# Конфиг
# --------------------------------------------------------------------------


class GrokConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.x.ai/v1/chat/completions"
    fast_model: str = "grok-4-fast"
    checker_model: str = "grok-4"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 1.0


class JitoConfig(BaseModel):
    enabled: bool = True
    block_engine_url: str = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"
    tip_lamports: int = 1_000_000


class SolanaConfig(BaseModel):
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    wallet_private_key: str = ""
    jito: JitoConfig = Field(default_factory=JitoConfig)


class DataConfig(BaseModel):
    api_key: str = ""
    rest_url: str = "https://frontend-api.pump.fun"
    ws_url: str = "wss://pumpportal.fun/api/data"
    request_timeout: float = 10.0


class RiskConfig(BaseModel):
    max_sol_per_trade: float = 0.5
    daily_loss_limit_sol: float = 2.0
    max_trades_per_day: int = 20
    max_open_positions: int = 3
    stop_loss_pct: float = 30.0
    stop_loss_poll_seconds: float = 15.0


class FilterConfig(BaseModel):
    min_unique_buyers: int = 5
    max_curve_progress: float = 0.40
    require_metadata: bool = True
    min_age_seconds: float = 120.0
    max_risk_score: float = 7.0
    min_total_score: float = 0.65


class ScoringWeights(BaseModel):
    audit: float = 0.30
    narrative: float = 0.25
    timing: float = 0.15
    metrics: float = 0.30


class ScoringConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    timing_cache_seconds: float = 900.0


class LoggingConfig(BaseModel):
    path: str = "logs/trades.jsonl"
    level: str = "INFO"


class Config(BaseModel):
    mode: Literal["dry-run", "live"] = "dry-run"
    grok: GrokConfig = Field(default_factory=GrokConfig)
    solana: SolanaConfig = Field(default_factory=SolanaConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> Config:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)
