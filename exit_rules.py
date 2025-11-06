# exit_rules.py
"""
Exit Rule Engine (extended)
New rule types added:
 - percent_of_portfolio: Suggest partial exit sizing (exit_size_pct) without closing position unless configured.
 - atr_stop: ATR-based fixed stop using recent OHLC series (requires set_ohlc_series).
 - atr_trailing: ATR-based trailing stop from observed peak (requires ATR calculation).
 - time_decay_take_profit: take-profit target that decays over time from initial -> final value.

How to use:
 - engine = ExitRuleEngine(entry_price=..., entry_time=...)
 - engine.set_ohlc_series(list_of_ohlc_dicts)  # required for ATR rules
 - engine.set_rules([...])  # include ExitRule entries
 - engine.on_price_tick(price, ts)
 - dec = engine.evaluate()
 - dec.exit_size_pct may be set for partial closes
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import math

# -------------------------
# Data models
# -------------------------
@dataclass
class ExitDecision:
    """Result returned when an exit is triggered."""
    exit_price: float
    exit_time: float
    reason: str  # e.g., 'take_profit', 'stop_loss', 'trailing_stop', 'time_exit', 'atr_stop', ...
    profit_pct: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)
    # for partial exits
    exit_size_pct: Optional[float] = None  # e.g., 50 -> exit 50% of position

@dataclass
class ExitRule:
    """
    A single exit rule. Compose multiple rules in a list and the engine will evaluate them
    in the order provided. The first rule that triggers yields the ExitDecision.

    Additional fields used by new rule types:
      - close_position: bool (default True). If False, rule suggests a partial exit without marking closed.
      - For ATR rules: 'atr_period' (int), 'multiplier' (float).
      - For time-decay: 'initial_pct', 'final_pct', 'decay_seconds'.
      - For percent_of_portfolio: 'size_pct' (float).
    """
    type: str
    value: float = 0.0
    multiple_mode: bool = False
    max_hold_seconds: Optional[int] = None
    enabled: bool = True
    name: Optional[str] = None
    # new fields:
    close_position: bool = True
    # ATR-specific
    atr_period: int = 14
    multiplier: float = 1.0
    # percent_of_portfolio
    size_pct: Optional[float] = None
    # time-decay TP
    initial_pct: Optional[float] = None
    final_pct: Optional[float] = None
    decay_seconds: Optional[int] = None

# -------------------------
# Engine
# -------------------------
class ExitRuleEngine:
    """
    Tracks trade lifecycle and evaluates ExitRules.
    New usage:
      engine = ExitRuleEngine(entry_price=..., entry_time=...)
      engine.set_ohlc_series( [ {"high":..,"low":..,"close":..}, ... ] )  # for ATR rules
      engine.set_rules([...])
      engine.on_price_tick(...)
      dec = engine.evaluate()
    """

    def __init__(self, entry_price: float, entry_time: Optional[float] = None):
        self.entry_price = float(entry_price)
        self.entry_time = float(entry_time or time.time())
        self.rules: List[ExitRule] = []
        # runtime state
        self.peak_price: float = self.entry_price
        self.trough_price: float = self.entry_price
        self.last_price: float = self.entry_price
        self.last_timestamp: float = self.entry_time
        self.closed = False

        # For ATR and history-based rules:
        # ohlc_list is a chronological list of dicts: {"high":float,"low":float,"close":float}
        self.ohlc_list: List[Dict[str, float]] = []

    def set_rules(self, rules: List[ExitRule]):
        self.rules = [r for r in rules if r.enabled]

    def set_ohlc_series(self, ohlc_list: List[Dict[str, float]]):
        """
        Provide a series of OHLC dicts (chronological) used for ATR computation.
        Each item: {"high":..., "low":..., "close":...}
        """
        self.ohlc_list = ohlc_list or []

    def on_price_tick(self, price: float, ts: Optional[float] = None):
        """Call this on each price update to update internal state (peak/trough)."""
        if self.closed:
            return
        ts = float(ts or time.time())
        price = float(price)
        self.last_price = price
        self.last_timestamp = ts
        if price > self.peak_price:
            self.peak_price = price
        if price < self.trough_price:
            self.trough_price = price

    def current_multiple(self) -> float:
        if self.entry_price == 0:
            return float("inf")
        return self.last_price / self.entry_price

    def current_profit_pct(self) -> float:
        return (self.last_price - self.entry_price) / self.entry_price * 100.0

    def time_held_seconds(self) -> float:
        return float(self.last_timestamp - self.entry_time)

    # -------------------------
    # ATR helpers
    # -------------------------
    def _compute_tr_list(self):
        """
        Return list of True Range (TR) values for self.ohlc_list.
        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        """
        tr_list = []
        prev_close = None
        for item in self.ohlc_list:
            h = float(item.get("high", 0.0))
            l = float(item.get("low", 0.0))
            c = float(item.get("close", 0.0))
            if prev_close is None:
                tr = h - l
            else:
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            tr_list.append(tr)
            prev_close = c
        return tr_list

    def compute_atr(self, period: int = 14) -> Optional[float]:
        """
        Compute a simple Wilder SMA ATR over the provided ohlc_list.
        Returns last ATR (float) or None if insufficient data.
        """
        if not self.ohlc_list:
            return None
        tr_list = self._compute_tr_list()
        if len(tr_list) < period:
            # not enough data; compute SMA over available
            return sum(tr_list) / len(tr_list) if tr_list else None
        # Wilder's smoothing: first ATR = SMA of first 'period' TRs, then ATR = (prev_atr*(period-1) + tr)/period iteratively.
        first_atr = sum(tr_list[:period]) / period
        atr = first_atr
        for tr in tr_list[period:]:
            atr = (atr * (period - 1) + tr) / period
        return float(atr)

    # -------------------------
    # Main evaluate loop
    # -------------------------
    def evaluate(self) -> Optional[ExitDecision]:
        """Evaluate all rules in configured order. Return first ExitDecision or None."""
        if self.closed:
            return None

        for rule in self.rules:
            if not rule.enabled:
                continue
            typ = rule.type
            dec = None
            if typ == "take_profit":
                dec = self._eval_take_profit(rule)
            elif typ == "stop_loss":
                dec = self._eval_stop_loss(rule)
            elif typ == "trailing_stop":
                dec = self._eval_trailing_stop(rule)
            elif typ == "time_exit":
                dec = self._eval_time_exit(rule)
            elif typ == "percent_of_portfolio":
                dec = self._eval_percent_of_portfolio(rule)
            elif typ == "atr_stop":
                dec = self._eval_atr_stop(rule)
            elif typ == "atr_trailing":
                dec = self._eval_atr_trailing(rule)
            elif typ == "time_decay_take_profit":
                dec = self._eval_time_decay_take_profit(rule)
            else:
                # unknown type: ignore
                continue

            if dec:
                # if rule requests to fully close position, mark closed
                if rule.close_position:
                    self.closed = True
                return dec

        return None

    # -------------------------
    # Rule evaluators (existing)
    # -------------------------
    def _eval_take_profit(self, rule: ExitRule) -> Optional[ExitDecision]:
        if rule.multiple_mode:
            target_multiple = float(rule.value)
            if self.current_multiple() >= target_multiple:
                profit_pct = (self.last_price / self.entry_price - 1.0) * 100.0
                return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="take_profit_multiple",
                                    profit_pct=profit_pct, info={"target_multiple": target_multiple})
        else:
            target_pct = float(rule.value)
            if self.current_profit_pct() >= target_pct:
                profit_pct = self.current_profit_pct()
                return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="take_profit_pct",
                                    profit_pct=profit_pct, info={"target_pct": target_pct})
        return None

    def _eval_stop_loss(self, rule: ExitRule) -> Optional[ExitDecision]:
        threshold_pct = float(rule.value)
        current_pct = self.current_profit_pct()
        if current_pct <= -abs(threshold_pct):
            profit_pct = current_pct
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="stop_loss",
                                profit_pct=profit_pct, info={"threshold_pct": threshold_pct})
        return None

    def _eval_trailing_stop(self, rule: ExitRule) -> Optional[ExitDecision]:
        trail_pct = float(rule.value)
        if self.peak_price <= 0:
            return None
        threshold_price = self.peak_price * (1.0 - trail_pct / 100.0)
        if self.last_price <= threshold_price:
            profit_pct = (self.last_price / self.entry_price - 1.0) * 100.0
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="trailing_stop",
                                profit_pct=profit_pct, info={"peak_price": self.peak_price, "trail_pct": trail_pct, "threshold_price": threshold_price})
        return None

    def _eval_time_exit(self, rule: ExitRule) -> Optional[ExitDecision]:
        max_hold = rule.max_hold_seconds if rule.max_hold_seconds is not None else rule.value
        if max_hold is None:
            return None
        if self.time_held_seconds() >= float(max_hold):
            profit_pct = self.current_profit_pct()
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="time_exit",
                                profit_pct=profit_pct, info={"max_hold_seconds": max_hold})
        return None

    # -------------------------
    # New rule evaluators
    # -------------------------
    def _eval_percent_of_portfolio(self, rule: ExitRule) -> Optional[ExitDecision]:
        """
        Suggest a partial exit of the position.
        rule.size_pct should be provided (0-100). If not set, uses rule.value as fallback.
        This will by default NOT close the entire position unless rule.close_position=True.
        """
        size = float(rule.size_pct) if rule.size_pct is not None else float(rule.value or 0)
        if size <= 0:
            return None
        # Trigger logic: percent_of_portfolio rule typically sits alongside a trigger condition.
        # If value is used as an activation threshold (e.g., a profit pct), interpret accordingly:
        # If rule.value > 0, treat it as profit_pct threshold to suggest partial exit.
        trigger_pct = float(rule.value or 0)
        if trigger_pct != 0:
            if self.current_profit_pct() >= trigger_pct:
                profit_pct = self.current_profit_pct()
                return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="percent_of_portfolio",
                                    profit_pct=profit_pct, info={"size_pct": size}, exit_size_pct=size)
            else:
                return None
        else:
            # If no trigger specified, default to always suggest (caller will decide how to apply)
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="percent_of_portfolio",
                                profit_pct=self.current_profit_pct(), info={"size_pct": size}, exit_size_pct=size)

    def _eval_atr_stop(self, rule: ExitRule) -> Optional[ExitDecision]:
        """
        ATR-based fixed stop relative to entry price:
          exit when last_price <= entry_price - ATR * multiplier
        requires set_ohlc_series to compute ATR
        """
        multiplier = float(rule.multiplier or rule.value or 1.0)
        period = int(rule.atr_period or 14)
        atr = self.compute_atr(period=period)
        if atr is None:
            return None
        threshold_price = self.entry_price - (atr * multiplier)
        if self.last_price <= threshold_price:
            profit_pct = (self.last_price / self.entry_price - 1.0) * 100.0
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="atr_stop",
                                profit_pct=profit_pct, info={"atr": atr, "multiplier": multiplier, "threshold_price": threshold_price})
        return None

    def _eval_atr_trailing(self, rule: ExitRule) -> Optional[ExitDecision]:
        """
        ATR-based trailing stop: exit when last_price <= peak_price - ATR * multiplier
        Requires ATR data (set_ohlc_series)
        """
        multiplier = float(rule.multiplier or rule.value or 1.0)
        period = int(rule.atr_period or 14)
        atr = self.compute_atr(period=period)
        if atr is None:
            return None
        threshold_price = self.peak_price - (atr * multiplier)
        if self.last_price <= threshold_price:
            profit_pct = (self.last_price / self.entry_price - 1.0) * 100.0
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="atr_trailing",
                                profit_pct=profit_pct, info={"atr": atr, "multiplier": multiplier, "peak_price": self.peak_price, "threshold_price": threshold_price})
        return None

    def _eval_time_decay_take_profit(self, rule: ExitRule) -> Optional[ExitDecision]:
        """
        Time-decay take profit: target decays linearly from initial_pct -> final_pct over decay_seconds.
        If current_profit_pct >= current_target => exit.
        rule.initial_pct (required), rule.final_pct (defaults to initial), rule.decay_seconds.
        """
        initial = float(rule.initial_pct) if rule.initial_pct is not None else float(rule.value or 0)
        final = float(rule.final_pct) if rule.final_pct is not None else initial
        decay_secs = int(rule.decay_seconds or 0)
        if initial == 0:
            return None
        elapsed = self.time_held_seconds()
        if decay_secs <= 0:
            current_target = final
        else:
            t = min(elapsed, decay_secs)
            # linear interpolation
            current_target = initial + (final - initial) * (t / decay_secs)
        if self.current_profit_pct() >= current_target:
            profit_pct = self.current_profit_pct()
            return ExitDecision(exit_price=self.last_price, exit_time=self.last_timestamp, reason="time_decay_take_profit",
                                profit_pct=profit_pct, info={"current_target_pct": current_target, "initial_pct": initial, "final_pct": final, "elapsed": elapsed, "decay_seconds": decay_secs})
        return None

    # -------------------------
    # Helpers
    # -------------------------
    def manual_close(self, price: float, ts: Optional[float] = None, reason: Optional[str] = "manual") -> ExitDecision:
        ts = ts or time.time()
        self.last_price = price
        self.last_timestamp = ts
        self.closed = True
        profit_pct = self.current_profit_pct()
        return ExitDecision(exit_price=price, exit_time=ts, reason=reason, profit_pct=profit_pct, info={})

# -------------------------
# Convenience factory parse helper (extended)
# -------------------------
def parse_rule_spec(spec: Dict[str, Any]) -> ExitRule:
    """
    Parse a simple dict spec into ExitRule.
    Supports new keys for ATR and time-decay and percent_of_portfolio.
    Examples:
      {"type":"atr_stop","multiplier":2.0,"atr_period":14}
      {"type":"atr_trailing","multiplier":3.0,"atr_period":14}
      {"type":"percent_of_portfolio","size_pct":50,"value":20}  # suggest 50% exit when profit >= 20%
      {"type":"time_decay_take_profit","initial_pct":100,"final_pct":50,"decay_seconds":3600}
    """
    typ = spec.get("type")
    value = spec.get("value", 0)
    multiple_mode = bool(spec.get("multiple_mode", False))
    max_hold = spec.get("max_hold_seconds", spec.get("max_hold", None))
    name = spec.get("name")
    close_position = bool(spec.get("close_position", True))
    atr_period = int(spec.get("atr_period", spec.get("atr_period", 14)))
    multiplier = float(spec.get("multiplier", spec.get("multiplier", 1.0)))
    size_pct = spec.get("size_pct", None)
    initial_pct = spec.get("initial_pct", None)
    final_pct = spec.get("final_pct", None)
    decay_seconds = spec.get("decay_seconds", None)

    return ExitRule(
        type=typ,
        value=float(value or 0),
        multiple_mode=multiple_mode,
        max_hold_seconds=max_hold,
        name=name,
        close_position=close_position,
        atr_period=atr_period,
        multiplier=multiplier,
        size_pct=size_pct,
        initial_pct=initial_pct,
        final_pct=final_pct,
        decay_seconds=decay_seconds
    )
