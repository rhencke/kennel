"""Tests for fido.atomic — AtomicReference[T] primitive."""

import threading
from dataclasses import dataclass

from frozendict import frozendict

from fido.atomic import AtomicReference
from fido.lens import Lens


@dataclass(frozen=True)
class _State:
    n: int


class TestAtomicReferenceGet:
    def test_get_returns_initial_value(self) -> None:
        ref: AtomicReference[int] = AtomicReference(42)
        assert ref.get() == 42

    def test_get_returns_same_object(self) -> None:
        obj = object()
        ref: AtomicReference[object] = AtomicReference(obj)
        assert ref.get() is obj


class TestAtomicReferenceSet:
    def test_set_replaces_value(self) -> None:
        ref: AtomicReference[int] = AtomicReference(1)
        ref.set(99)
        assert ref.get() == 99

    def test_set_replaces_with_new_object(self) -> None:
        a = _State(0)
        b = _State(1)
        ref: AtomicReference[_State] = AtomicReference(a)
        ref.set(b)
        assert ref.get() is b


class TestAtomicReferenceCompareAndSet:
    def test_succeeds_when_expected_matches(self) -> None:
        a = _State(0)
        b = _State(1)
        ref: AtomicReference[_State] = AtomicReference(a)
        success, value = ref.compare_and_set(a, b)
        assert success is True
        assert value is b
        assert ref.get() is b

    def test_fails_when_expected_differs(self) -> None:
        a = _State(0)
        other = _State(0)  # equal value, different identity
        b = _State(1)
        ref: AtomicReference[_State] = AtomicReference(a)
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
        ref: AtomicReference[tuple[int, ...]] = AtomicReference(x)
        success, current = ref.compare_and_set(y, new)
        assert success is False
        assert current is x  # returned the actual current reference
        assert ref.get() is x

    def test_leaves_value_unchanged_on_failure(self) -> None:
        a = _State(10)
        wrong = _State(99)
        new = _State(20)
        ref: AtomicReference[_State] = AtomicReference(a)
        success, current = ref.compare_and_set(wrong, new)
        assert success is False
        assert current is a
        assert ref.get() is a


class TestAtomicReferenceUpdate:
    def test_applies_function_to_current_value(self) -> None:
        ref: AtomicReference[_State] = AtomicReference(_State(10))
        result = ref.update(lambda s: _State(s.n + 5))
        assert result == _State(15)
        assert ref.get() == _State(15)

    def test_returns_installed_value(self) -> None:
        ref: AtomicReference[_State] = AtomicReference(_State(0))
        installed = ref.update(lambda s: _State(s.n + 1))
        assert installed is ref.get()

    def test_retries_after_concurrent_modification(self) -> None:
        """update() retries when a writer sneaks in between get() and CAS."""
        ref: AtomicReference[_State] = AtomicReference(_State(0))
        injected = [False]

        def fn(s: _State) -> _State:
            if not injected[0]:
                injected[0] = True
                # Race: overwrite the ref before update() can CAS.
                ref.set(_State(100))
            return _State(s.n + 1)

        result = ref.update(fn)
        # Iteration 1: s=State(0), ref sneaked to State(100), returns State(1).
        #   CAS(State(0), State(1)) fails — identities differ.
        # Iteration 2: s=State(100), returns State(101).
        #   CAS(State(100), State(101)) succeeds.
        assert result == _State(101)
        assert ref.get() == _State(101)

    def test_concurrent_updates_all_applied(self) -> None:
        """Many concurrent update() calls; no increments are lost."""
        ref: AtomicReference[_State] = AtomicReference(_State(0))
        n_threads = 20
        increments_per_thread = 50

        def run() -> None:
            for _ in range(increments_per_thread):
                ref.update(lambda s: _State(s.n + 1))

        threads = [threading.Thread(target=run) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ref.get() == _State(n_threads * increments_per_thread)


@dataclass(frozen=True)
class _Item:
    val: int


@dataclass(frozen=True)
class _Container:
    items: frozendict[str, _Item]


class TestAtomicReferenceLensUpdate:
    def test_installs_value_at_path(self) -> None:
        state = _Container(items=frozendict({"a": _Item(1)}))
        ref: AtomicReference[_Container] = AtomicReference(state)
        new_item = _Item(99)
        result = ref.lens_update(lambda root: root.items["a"], new_item)
        assert result.items["a"] is new_item
        assert ref.get().items["a"] is new_item

    def test_inserts_new_key(self) -> None:
        ref: AtomicReference[_Container] = AtomicReference(
            _Container(items=frozendict())
        )
        new_item = _Item(7)
        ref.lens_update(lambda root: root.items["new"], new_item)
        assert ref.get().items["new"] is new_item

    def test_returns_installed_root(self) -> None:
        ref: AtomicReference[_Container] = AtomicReference(
            _Container(items=frozendict({"x": _Item(0)}))
        )
        result = ref.lens_update(lambda root: root.items["x"], _Item(5))
        assert result is ref.get()

    def test_selector_receives_lens(self) -> None:
        """selector is called with a Lens, not the raw value."""
        received: list[object] = []

        def selector(root: Lens[_Container]) -> Lens[_Container]:
            received.append(root)
            return root.items["a"]

        ref: AtomicReference[_Container] = AtomicReference(
            _Container(items=frozendict({"a": _Item(1)}))
        )
        ref.lens_update(selector, _Item(2))
        assert isinstance(received[0], Lens)

    def test_preserves_sibling_keys(self) -> None:
        a = _Item(1)
        b = _Item(2)
        ref: AtomicReference[_Container] = AtomicReference(
            _Container(items=frozendict({"a": a, "b": b}))
        )
        ref.lens_update(lambda root: root.items["a"], _Item(99))
        assert ref.get().items["b"] is b
