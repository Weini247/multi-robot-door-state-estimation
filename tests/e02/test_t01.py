"""ARGUS-E02-T01: event ordering, causality and call-order independent randomness."""

from __future__ import annotations

from itertools import permutations

import pytest

from argus.core.events import KeyedRandom, Phase, SimulationClock, SimulationEvent


def drain(clock: SimulationClock) -> list[str]:
    return [clock.pop().event_id for _ in range(len(clock))]


EVENTS = [
    SimulationEvent(2, Phase.POLICY, "policy/r1"),
    SimulationEvent(2, Phase.POLICY, "policy/r0"),
    SimulationEvent(1, Phase.SENSOR, "sensor/r0/slot0"),
    SimulationEvent(2, Phase.WORLD, "world/2"),
    SimulationEvent(1, Phase.DELIVERY, "delivery/m7"),
]
EXPECTED = ["delivery/m7", "sensor/r0/slot0", "world/2", "policy/r0", "policy/r1"]


def test_every_insertion_permutation_yields_the_same_order() -> None:
    for order in permutations(EVENTS):
        clock = SimulationClock()
        for event in order:
            clock.schedule(event)
        assert drain(clock) == EXPECTED


def test_phases_run_in_architecture_order() -> None:
    assert [p.name for p in sorted(Phase)] == [
        "WORLD",
        "DELIVERY",
        "SENSOR",
        "BELIEF",
        "POLICY",
        "EVALUATE",
    ]


def test_duplicate_event_id_rejected_even_after_pop() -> None:
    clock = SimulationClock()
    clock.schedule(SimulationEvent(0, Phase.WORLD, "world/0"))
    with pytest.raises(ValueError, match="duplicate"):
        clock.schedule(SimulationEvent(5, Phase.WORLD, "world/0"))
    clock.pop()
    with pytest.raises(ValueError, match="duplicate"):
        clock.schedule(SimulationEvent(9, Phase.WORLD, "world/0"))


def test_backward_and_same_key_events_rejected() -> None:
    clock = SimulationClock()
    clock.schedule(SimulationEvent(3, Phase.SENSOR, "sensor/b"))
    clock.pop()
    for late in (
        SimulationEvent(2, Phase.EVALUATE, "late/tick"),
        SimulationEvent(3, Phase.DELIVERY, "late/phase"),
        SimulationEvent(3, Phase.SENSOR, "sensor/a"),
    ):
        with pytest.raises(ValueError, match="not after"):
            clock.schedule(late)
    clock.schedule(SimulationEvent(3, Phase.SENSOR, "sensor/c"))  # strictly later id is fine
    clock.schedule(SimulationEvent(3, Phase.BELIEF, "belief/a"))
    assert drain(clock) == ["sensor/c", "belief/a"]
    assert clock.tick == 3


@pytest.mark.parametrize(
    "event",
    [
        SimulationEvent(-1, Phase.WORLD, "neg"),
        SimulationEvent(1.0, Phase.WORLD, "float"),  # type: ignore[arg-type]
        SimulationEvent(1, Phase.WORLD, ""),
        SimulationEvent(1, Phase.WORLD, "nan", {"x": float("nan")}),
        SimulationEvent(1, Phase.WORLD, "obj", {"x": object()}),
    ],
)
def test_invalid_events_rejected(event: SimulationEvent) -> None:
    with pytest.raises((ValueError, TypeError)):
        SimulationClock().schedule(event)


def test_restore_preserves_order_ids_and_causality() -> None:
    clock = SimulationClock()
    for event in EVENTS:
        clock.schedule(event)
    first = clock.pop()
    restored = SimulationClock.restore(clock.last_key, clock.used_ids(), clock.pending())
    assert restored.last_key == first.key
    with pytest.raises(ValueError):
        restored.schedule(SimulationEvent(9, Phase.WORLD, "policy/r1"))
    assert drain(restored) == EXPECTED[1:]


def test_additional_sensor_calls_do_not_change_world_keys() -> None:
    rng = KeyedRandom(7)
    keys = [(edge, tick) for edge in ("door-a", "door-b") for tick in range(50)]
    plain = [rng.uniform("world.passage", e, t) for e, t in keys]
    noisy = []
    for i, (edge, tick) in enumerate(keys):
        for extra in range(i % 4):
            rng.uniform("sensor.private", "r1", edge, tick, extra)
        noisy.append(rng.uniform("world.passage", edge, tick))
    assert noisy == plain


def test_domains_seeds_and_keys_are_separated() -> None:
    a = KeyedRandom(7)
    assert a.uniform("world.passage", "door", 3) != a.uniform("sensor.private", "door", 3)
    assert a.uniform("world.passage", "door", 3) != KeyedRandom(8).uniform(
        "world.passage", "door", 3
    )
    assert a.uniform("world.passage", "door", 3) != a.uniform("world.passage", "door", 4)
    # Keys are typed: the int 3 and the string "3" are different keys.
    assert a.uniform("world.passage", "door", 3) != a.uniform("world.passage", "door", "3")


def test_golden_values_pin_rng_schema_v1() -> None:
    rng = KeyedRandom(7)
    assert rng.uniform("world.passage", "door-a", 0) == pytest.approx(GOLDEN[0], abs=0)
    assert rng.uniform("sensor.private", "r1", "door-a", 12, 0) == pytest.approx(GOLDEN[1], abs=0)
    assert KeyedRandom(0).uniform("channel.loss", "r1->r2", 99) == pytest.approx(GOLDEN[2], abs=0)


# Versioned golden values of argus-rng-v1; changing them requires a new RNG schema.
GOLDEN = (0.18960346003793582, 0.638163357679127, 0.39916511495199625)


def test_schema_matches_blueprint_reference_keyed_uniform() -> None:
    from reference.events import keyed_uniform

    rng = KeyedRandom(7)
    for keys in (("room1", 9), ("door-a", 0), ("r1", "door-a", 12, 0)):
        assert rng.uniform("sensor", *keys) == keyed_uniform(7, "sensor", *keys)


def test_uniform_range_and_moments_on_fixed_keys() -> None:
    rng = KeyedRandom(123)
    values = [rng.uniform("check", i) for i in range(20000)]
    assert all(0.0 <= v < 1.0 for v in values)
    mean = sum(values) / len(values)
    # Deterministic keys, so this cannot flake; 5 sigma of the U[0,1) mean.
    assert abs(mean - 0.5) < 5 * (1 / 12) ** 0.5 / len(values) ** 0.5


def test_bernoulli_matches_uniform_threshold_and_validates() -> None:
    rng = KeyedRandom(5)
    for i in range(100):
        assert rng.bernoulli(0.3, "b", i) == int(rng.uniform("b", i) < 0.3)
    assert rng.bernoulli(0.0, "b", 1) == 0 and rng.bernoulli(1.0, "b", 1) == 1
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            rng.bernoulli(bad, "b", 1)
    for bad_key in (1.5, True, None):
        with pytest.raises(ValueError):
            rng.uniform("b", bad_key)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        KeyedRandom(-1)
