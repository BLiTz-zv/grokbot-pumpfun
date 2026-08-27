"""Агент 3: момент рынка.

Оценивает не токен, а фон: настроение, идёт ли мем-сезон, какие сейчас
объёмы на pump.fun, нет ли аномалий. Ответ одинаков для всех токенов в
пределах окна, поэтому результат кэшируется на `timing_cache_seconds`
(по умолчанию 15 минут) — иначе каждый лонч оплачивал бы один и тот же вывод.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, ClassVar

from ..models import Config, TimingResult
from .base import JSON_ONLY, GrokAgent

TIMING_PROMPT = f"""Ты — аналитик рыночного режима для мемкоинов Solana.
Оцениваешь не конкретный токен, а фон, на котором он запускается.

ВАЖНО про твои данные. Тебе дают не рыночную сводку, а результаты
собственного наблюдения бота за последним окном: сколько лончей в минуту
прошло через сокет, какая доля дожила до разбора, сколько SOL они успевают
собрать, чем кончились последние сделки самого бота. Это всё, что известно
достоверно. Внешних котировок, новостей и данных по BTC у тебя нет — не
делай вид, что есть, и не выдумывай событий.

Поля, которые приходят:
  лончей_в_минуту, лончей_в_окне — интенсивность потока новых токенов;
  доля_доживших_до_разбора — какая часть лончей прошла базовый фильтр;
  медиана_sol_в_кривой — сколько типичный лонч успевает собрать;
  доля_прибыльных, медиана_результата_pct, доля_сливов — исходы последних
  сделок бота, если они были;
  час_utc — время суток, ликвидность мемкоинов заметно от него зависит;
  данных_мало — поток почти пуст, выводы ненадёжны.

Дай три оценки от 0.0 до 1.0 и список аномалий:
- market_sentiment: общее настроение. Высокое, когда лончи набирают
  ликвидность и сделки закрываются в плюс; низкое, когда доля сливов
  растёт, а результаты уходят в минус.
- meme_season: доходят ли свежие лончи до заметных объёмов. Опирайся на
  медиану SOL в кривой и на долю доживших до разбора, а не на ощущения.
- volume_level: интенсивность потока относительно того, что обычно бывает
  в этот час UTC.
- anomalies: короткие метки того, что ломает нормальную торговлю —
  "поток_иссяк", "сливы_подряд", "ночной_штиль", "лончи_не_набирают",
  "данных_мало". Пустой список, если фон обычный.

Правила:
- при данных_мало = true ставь оценки не выше 0.5 и добавляй метку
  "данных_мало": торговать вслепую хуже, чем пропустить;
- пустая история сделок — это не хороший фон, а отсутствие сведений;
- не объясняй числа внешними причинами, которых не видел.

Формат ответа:
{{
  "market_sentiment": 0.0-1.0,
  "meme_season": 0.0-1.0,
  "volume_level": 0.0-1.0,
  "anomalies": ["метки"],
  "reasoning": "2-3 предложения со ссылкой на конкретные числа из входа"
}}

{JSON_ONLY}"""


class TimingAgent(GrokAgent):
    name: ClassVar[str] = "timing"
    prompt: ClassVar[str] = TIMING_PROMPT
    result_model: ClassVar[type] = TimingResult

    def __init__(
        self,
        config: Config,
        client: Any | None = None,
        ops: Any | None = None,
    ) -> None:
        super().__init__(config, client, ops)
        self.cache_seconds = config.scoring.timing_cache_seconds
        self._cached: TimingResult | None = None
        self._lock = asyncio.Lock()

    def build_user_message(self, market_snapshot: dict[str, Any] | None = None) -> str:
        payload = {
            "спрошено_unix": int(time.time()),
            "наблюдения": market_snapshot or {},
        }
        return json.dumps(payload, ensure_ascii=False)

    def fallback(self, reason: str) -> TimingResult:
        return TimingResult.pessimistic(reason)

    # -- кэш ---------------------------------------------------------------

    def cache_is_fresh(self, now: float | None = None) -> bool:
        if self._cached is None:
            return False
        now = now or time.time()
        return (now - self._cached.fetched_at) < self.cache_seconds

    async def get(self, market_snapshot: dict[str, Any] | None = None) -> TimingResult:
        """Свежая оценка рынка: из кэша или новым вызовом.

        Лок нужен, чтобы пачка токенов, подошедших одновременно, не устроила
        три параллельных одинаковых запроса.
        """
        if self.cache_is_fresh():
            return self._cached  # type: ignore[return-value]
        async with self._lock:
            if self.cache_is_fresh():
                return self._cached  # type: ignore[return-value]
            result: TimingResult = await self.run(market_snapshot)
            result.fetched_at = time.time()
            # Пессимистичный фолбэк не кэшируем: сбой не должен блокировать
            # рынок на все 15 минут.
            if "agent_failure" not in result.anomalies:
                self._cached = result
            return result

    def invalidate(self) -> None:
        self._cached = None
