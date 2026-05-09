"""Tests for fido.atomic — create_atomic factory and _AtomicReference internals."""

import threading
from dataclasses import dataclass

from frozendict import frozendict

from fido.atomic import (
    AtomicReader,
    AtomicUpdater,
    _AtomicReference,  # noqa: PLC2701
    create_atomic,
)
from fido.lens import Lens


@dataclass(frozen=True)
class _State:
    n: int


class TestCreateAtomic:
    """Tests for the public create_atomic factory."""

    def test_returns_reader_updater_tuple(self) -> None:
        reader, updater = create_atomic(42)
        assert reader.get() == 42

    def test_reader_satisfies_atomic_reader_protocol(self) -> None:
        reader, _ = create_atomic(7)
        ref: AtomicReader[int] = reader
        assert ref.get() == 7

    def test_updater_satisfies_atomic_updater_protocol(self) -> None:
        state = _Container(items=frozendict({"x": _Item(0)}))
        _, updater = create_atomic(state)
        ref: AtomicUpdater[_Container] = updater
        ref.update(lambda root: root.items["x"], _Item(9))

    def test_reader_and_updater_share_state(self) -> None:
        reader, updater = create_atomic(_Container(items=frozendict({"k": _Item(0)})))
        updater.update(lambda root: root.items["k"], _Item(99))
        assert reader.get().items["k"] == _Item(99)

    def test_reader_has_no_update_attribute(self) -> None:
        reader, _ = create_atomic(0)
        # AtomicReader protocol exposes only .get() — no .update()
        assert not hasattr(AtomicReader, "update")

    def test_updater_has_no_get_attribute(self) -> None:
        _, updater = create_atomic(0)
        # AtomicUpdater protocol exposes only .update() — no .get()
        assert not hasattr(AtomicUpdater, "get")


class TestAtomicReferenceGet:
    def test_get_returns_initial_value(self) -> None:
        ref: _AtomicReference[int] = _AtomicReference(42)
        assert ref.get() == 42

    def test_get_returns_same_object(self) -> None:
        obj = object()
        ref: _AtomicReference[object] = _AtomicReference(obj)
        assert ref.get() is obj


class TestAtomicReferenceSet:
    def test_set_replaces_value(self) -> None:
        ref: _AtomicReference[int] = _AtomicReference(1)
        ref.set(99)
        assert ref.get() == 99

    def test_set_replaces_with_new_object(self) -> None:
        a = _State(0)
        b = _State(1)
        ref: _AtomicReference[_State] = _AtomicReference(a)
        ref.set(b)
        assert ref.get() is b


class TestAtomicReferenceCompareAndSet:
    def test_succeeds_when_expected_matches(self) -> None:
        a = _State(0)
        b = _State(1)
        ref: _AtomicReference[_State] = _AtomicReference(a)
        success, value = ref.compare_and_set(a, b)
        assert success is True
        assert value is b
        assert ref.get() is b

    def test_fails_when_expected_differs(self) -> None:
        a = _State(0)
        other = _State(0)  # equal value, different identity
        b = _State(1)
        ref: _AtomicReference[_State] = _AtomicReference(a)
        success, current = ref.compare_and_set(other, b)
        assert success is False
        assert current is a  # returned current reference, not the rejected new value
        assert ref.get() is a  # unchanged

    def test_uses_identity_not_equality(self) -> None:
        # Two equal tuples — CAS should fail because they are distinct objects.
        # Use tuple() at runtime to avoid CPython constant-folding identical
        # literals to the same object.
        x: tuple[int, ...] = tuple(range(3))
        y: tuple[int, ...] = tuple(range(3))
        assert x == y
        assert x is not y  # distinct objects despite equal values
        new: tuple[int, ...] = tuple(range(4, 7))
        ref: _AtomicReference[tuple[int, ...]] = _AtomicReference(x)
        success, current = ref.compare_and_set(y, new)
        assert success is False
        assert current is x  # returned the actual current reference
        assert ref.get() is x

    def test_leaves_value_unchanged_on_failure(self) -> None:
        a = _State(10)
        wrong = _State(99)
        new = _State(20)
        ref: _AtomicReference[_State] = _AtomicReference(a)
        success, current = ref.compare_and_set(wrong, new)
        assert success is False
        assert current is a
        assert ref.get() is a


@dataclass(frozen=True)
class _Item:
    val: int


@dataclass(frozen=True)
class _Container:
    items: frozendict[str, _Item]


class TestAtomicReferenceUpdate:
    def test_installs_value_at_path(self) -> None:
        state = _Container(items=frozendict({"a": _Item(1)}))
        ref: _AtomicReference[_Container] = _AtomicReference(state)
        new_item = _Item(99)
        result = ref.update(lambda root: root.items["a"], new_item)
        assert result.items["a"] is new_item
        assert ref.get().items["a"] is new_item

    def test_inserts_new_key(self) -> None:
        ref: _AtomicReference[_Container] = _AtomicReference(
            _Container(items=frozendict())
        )
        new_item = _Item(7)
        ref.update(lambda root: root.items["new"], new_item)
        assert ref.get().items["new"] is new_item

    def test_returns_installed_root(self) -> None:
        ref: _AtomicReference[_Container] = _AtomicReference(
            _Container(items=frozendict({"x": _Item(0)}))
        )
        result = ref.update(lambda root: root.items["x"], _Item(5))
        assert result is ref.get()

    def test_selector_receives_lens(self) -> None:
        """selector is called with a Lens, not the raw value."""
        received: list[object] = []

        def selector(root: Lens[_Container]) -> Lens[_Container]:
            received.append(root)
            return root.items["a"]

        ref: _AtomicReference[_Container] = _AtomicReference(
            _Container(items=frozendict({"a": _Item(1)}))
        )
        ref.update(selector, _Item(2))
        assert isinstance(received[0], Lens)

    def test_preserves_sibling_keys(self) -> None:
        a = _Item(1)
        b = _Item(2)
        ref: _AtomicReference[_Container] = _AtomicReference(
            _Container(items=frozendict({"a": a, "b": b}))
        )
        ref.update(lambda root: root.items["a"], _Item(99))
        assert ref.get().items["b"] is b

    def test_retries_after_concurrent_modification(self) -> None:
        """update() retries when a concurrent writer sneaks in between get() and CAS."""
        ref: _AtomicReference[_Container] = _AtomicReference(
            _Container(items=frozendict({"a": _Item(0)}))
        )
        injected = [False]

        def selector(root: Lens[_Container]) -> Lens[_Container]:
            if not injected[0]:
                injected[0] = True
                # Race: overwrite the ref before update() can CAS.
                ref.set(_Container(items=frozendict({"a": _Item(100)})))
            return root.items["a"]

        result = ref.update(selector, _Item(42))
        # Iteration 1: selector runs on old root (a=0), sneaks in a=100,
        #   produces root with a=42.  CAS fails because ref changed.
        # Iteration 2: selector runs on new root (a=100), produces root
        #   with a=42.  CAS succeeds.
        assert result.items["a"] == _Item(42)
        assert ref.get().items["a"] == _Item(42)

    def test_concurrent_updates_no_lost_writes(self) -> None:
        """Many concurrent update() calls on disjoint keys; no writes lost."""
        n_threads = 20
        items = frozendict({f"k{i}": _Item(0) for i in range(n_threads)})
        ref: _AtomicReference[_Container] = _AtomicReference(_Container(items=items))

        def run(key: str) -> None:
            for v in range(1, 51):
                ref.update(lambda root: root.items[key], _Item(v))

        threads = [
            threading.Thread(target=run, args=(f"k{i}",)) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = ref.get()
        for i in range(n_threads):
            assert final.items[f"k{i}"] == _Item(50)
