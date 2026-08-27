"""Агент 4: адверсариальная проверка.

Последний рубеж перед деньгами и единственный, кто работает на сильной
модели. Ему запрещено искать причины купить: он получает выводы всех
предыдущих агентов и ищет, где они противоречат друг другу и что они
пропустили.

approve: false — это нормальный, ожидаемый исход. Ошибка вызова тоже
превращается в approve: false.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from ..curve import buy_quote
from ..models import Analysis, CheckerResult
from .base import JSON_ONLY, GrokAgent

CHECKER_PROMPT = f"""Ты — риск-офицер, который подписывает или блокирует
покупку мемкоина. Твоя работа — НЕ найти причины купить. Твоя работа —
найти причины НЕ покупать.

Тебе дают полный разбор: метрики, вывод аудитора кошельков, оценку
мем-потенциала, состояние рынка и итоговый скоринг.

Ищи:
1. Противоречия между сигналами. Высокий мем-потенциал при низкой органике
   покупателей. Хорошая кривая при концентрации у топ-5. Сильный скоринг,
   собранный одним компонентом при провале остальных.
2. Красные флаги, которые предыдущие агенты не отметили или недооценили.
3. Слабую доказательную базу: низкий confidence аудитора, мало сделок,
   отсутствие данных при уверенных выводах.
4. Рыночный фон, при котором даже хороший токен не поедет.
5. Экономику самой сделки. В блоке "план" лежит то, что реально
   произойдёт: размер заявки, во что обойдётся вход и немедленный выход
   (стоимость_круга_pct), насколько своя же заявка сдвинет цену
   (влияние_на_цену_pct), и где стоят выходы. Если стоимость круга
   сопоставима с движением, ради которого затевается сделка, или заявка
   двигает цену на проценты — это причина отказать, даже когда сам токен
   выглядит прилично.

Правила решения:
- Сомневаешься — approve: false.
- Любой сработавший флаг аудитора при organic_buyer_share ниже 0.5 —
  approve: false.
- Не одобряй на основании одного сильного компонента.

Формат ответа:
{{
  "approve": true|false,
  "reason": "одно-два предложения, главная причина решения",
  "flags": ["короткие метки найденных проблем"],
  "confidence": 0.0-1.0
}}

{JSON_ONLY}"""


class CheckerAgent(GrokAgent):
    name: ClassVar[str] = "checker"
    version: ClassVar[str] = "checker-2"
    prompt: ClassVar[str] = CHECKER_PROMPT
    result_model: ClassVar[type] = CheckerResult
    use_checker_model: ClassVar[bool] = True

    def build_user_message(self, analysis: Analysis) -> str:
        token = analysis.token
        payload = {
            "token": {
                "mint": token.mint,
                "name": token.name,
                "symbol": token.symbol,
                "description": token.description,
                "links": {
                    "twitter": token.twitter,
                    "telegram": token.telegram,
                    "website": token.website,
                },
                "age_seconds": round(token.age_seconds),
                "unique_buyers": token.unique_buyers,
                "curve_progress": round(token.curve_progress, 4),
            },
            "metrics": analysis.metrics.model_dump(),
            "auditor": analysis.audit.model_dump() if analysis.audit else None,
            "narrative": analysis.narrative.model_dump() if analysis.narrative else None,
            "timing": analysis.timing.model_dump() if analysis.timing else None,
            "scores": analysis.scores.model_dump(),
            "план": self._plan(analysis),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _plan(self, analysis: Analysis) -> dict[str, Any]:
        """Экономика сделки, которую собираемся совершить."""
        risk = self.config.risk
        plan: dict[str, Any] = {
            "размер_sol": round(analysis.plan.size_sol, 6) if analysis.plan else None,
            "одобрено_риском": analysis.plan.approved if analysis.plan else None,
            "стоимость_круга_pct": analysis.metrics.round_trip_cost_pct,
            "ликвидность_кривой_sol": analysis.metrics.curve_liquidity_sol,
            "take_profit_pct": risk.take_profit_pct,
            "stop_loss_pct": risk.stop_loss_pct,
            "лимит_удержания_мин": round(risk.max_hold_seconds / 60, 1),
        }
        if analysis.curve is not None and analysis.plan is not None:
            quote = buy_quote(analysis.curve, analysis.plan.size_sol,
                              self.config.market.trade_fee_pct)
            if quote.ok:
                plan["влияние_на_цену_pct"] = round(quote.impact_pct, 3)
        return plan

    def fallback(self, reason: str) -> CheckerResult:
        return CheckerResult.pessimistic(reason)
