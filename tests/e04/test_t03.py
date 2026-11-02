"""ARGUS-E04-T03: Gilbert-Elliott loss with an exogenous channel state."""

from __future__ import annotations

import math

import pytest

from argus.communication.loss import BAD, GilbertElliott
from argus.communication.messages import originate
from argus.communication.queue import Channel, ChannelConfig
from argus.core.events import KeyedRandom
from argus.core.types import EvidenceId, Observation

A, B, TICK = 0.02, 0.2, 0.1  # pilot defaults


def ge(eps_g: float = 0.01, eps_b: float = 0.5, seed: int = 1, **kw: object) -> GilbertElliott:
    values: dict[str, object] = dict(
        rate_good_to_bad=A,
        rate_bad_to_good=B,
        eps_good=eps_g,
        eps_bad=eps_b,
        tick_seconds=TICK,
        rng=KeyedRandom(seed),
        scenario_id="ge-fixture",
    )
    values.update(kw)
    return GilbertElliott(**values)  # type: ignore[arg-type]


LINK = ("r0", "r1")


def test_zero_loss_delivers_everything_and_full_loss_nothing() -> None:
    for eps, expected in ((0.0, False), (1.0, True)):
        model = ge(eps, eps)
        assert {model.drops(LINK, t, f"m{t}") for t in range(500)} == {expected}


def test_stationary_drop_rate_is_analytic() -> None:
    p_bad = A / (A + B)
    expected = 0.01 * (1 - p_bad) + 0.5 * p_bad
    assert ge().stationary_drop_rate() == pytest.approx(expected, abs=1e-15)
    # Empirical check over independent seeds (fixed keys; 5 standard errors set beforehand).
    n = 6000
    drops = sum(ge(seed=s).drops(LINK, 0, "m") for s in range(n))
    assert abs(drops / n - expected) < 5 * math.sqrt(expected * (1 - expected) / n)
    bad = sum(ge(seed=s).state(LINK, 0) == BAD for s in range(n))
    assert abs(bad / n - p_bad) < 5 * math.sqrt(p_bad * (1 - p_bad) / n)


def test_bad_state_is_bursty_with_the_ctmc_transition() -> None:
    model = ge(seed=3)
    states = [model.state(LINK, t) for t in range(60000)]
    stay_bad = [b for a, b in zip(states, states[1:], strict=False) if a == BAD]
    p_stay = model.model().transition(TICK)[1][1]
    assert abs(sum(stay_bad) / len(stay_bad) - p_stay) < 5 * math.sqrt(
        p_stay * (1 - p_stay) / len(stay_bad)
    )
    assert p_stay > 0.97  # bursts last many ticks, unlike independent loss


def test_additional_sends_do_not_shift_the_latent_channel_field() -> None:
    quiet, busy = ge(seed=5), ge(seed=5)
    for t in range(300):
        for k in range(t % 5):
            busy.drops(LINK, t, f"extra-{t}-{k}")
            busy.drops(("r1", "r2"), t, f"x-{k}")
    assert [quiet.state(LINK, t) for t in range(300)] == [busy.state(LINK, t) for t in range(300)]
    assert quiet.drops(LINK, 77, "m") == busy.drops(LINK, 77, "m")


def test_links_have_independent_states() -> None:
    model = ge(seed=2)
    a = [model.state(("r0", "r1"), t) for t in range(2000)]
    b = [model.state(("r1", "r0"), t) for t in range(2000)]
    assert a != b


def test_zero_rates_require_and_keep_the_initial_state() -> None:
    with pytest.raises(ValueError):
        ge(rate_good_to_bad=0.0, rate_bad_to_good=0.0)
    stuck = ge(rate_good_to_bad=0.0, rate_bad_to_good=0.0, initial_bad=1.0)
    assert {stuck.state(LINK, t) for t in range(100)} == {BAD}
    assert stuck.stationary_drop_rate() == 0.5


def test_channel_uses_the_loss_model_at_transmission_start() -> None:
    def msg(seq: int, tick: int):  # noqa: ANN202
        obs = Observation(EvidenceId("ep", "r0", seq), tick, "door", 1, (0.0, 0.0, 0.0), "m")
        return originate(obs, "r1", tick)

    config = ChannelConfig(0.1, 1e6, 10, 1e9, 0, 10**6)
    model = ge(0.0, 1.0, seed=4)
    channel = Channel(config, model)
    for t in range(200):
        channel.submit(msg(t, t), t)
    for attempt in channel.attempts:
        assert attempt.dropped == (model.state(attempt.link, attempt.start) == BAD)
    assert channel.stats.dropped_bytes["r0"] == sum(a.size for a in channel.attempts if a.dropped)


@pytest.mark.parametrize(
    "bad", [{"eps_good": -0.1}, {"eps_bad": 1.5}, {"tick_seconds": 0.0}, {"rate_good_to_bad": -1.0}]
)
def test_invalid_parameters_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ge(**bad)
