"""Lens — path-recording updater for frozen dataclass trees.

A :class:`Lens` wraps a frozen root value, records attribute and item
accesses as the caller navigates into nested structure, then reconstructs
the entire path of frozen copies when :meth:`~Lens.set` is called with
the new leaf value.

Frozen dataclass nodes are reconstructed via :func:`dataclasses.replace`.
Frozen mapping nodes (e.g. :class:`frozendict.frozendict`) are
reconstructed via ``type(parent)({**parent, key: new_value})``.

Typical usage with :class:`~fido.atomic.AtomicReference`::

    ref.update(lambda root: Lens(root).repos[name].set(new_repo_state))
"""

import dataclasses
from typing import Any, Generic, TypeVar

_R = TypeVar("_R")

# Step kinds stored in the path tuple.
_ATTR = "attr"
_ITEM = "item"


class Lens(Generic[_R]):
    """Path-recording proxy for functional updates to frozen dataclass trees.

    Each ``.attr`` or ``[key]`` access appends a step to the internal path
    without touching the original structure.  :meth:`set` walks the path
    forward to collect intermediate values, then backwards to produce
    updated frozen copies at every level — returning a new root of the
    same type.

    The proxy is deliberately minimal: ``__getattr__`` records attribute
    steps, ``__getitem__`` records key steps, and ``set`` is the only
    terminal operation.  This means a field literally named ``set`` on a
    target dataclass would shadow the method — acceptable for this
    codebase, where no frozen dataclass has such a field.
    """

    __slots__ = ("_root", "_path")

    def __init__(self, root: _R, _path: tuple[tuple[str, object], ...] = ()) -> None:
        self._root = root
        self._path = _path

    def __getattr__(self, name: str) -> "Lens[_R]":
        return Lens(self._root, self._path + ((_ATTR, name),))

    def __getitem__(self, key: object) -> "Lens[_R]":
        return Lens(self._root, self._path + ((_ITEM, key),))

    def set(self, value: object) -> _R:
        """Return a new root with *value* installed at the recorded path.

        Walks the recorded path forward to collect the intermediate values
        from the current root, then backwards — using
        :func:`dataclasses.replace` for dataclass nodes and mapping
        reconstruction for item-accessed nodes — to produce the updated
        root.
        """
        if not self._path:
            return value  # type: ignore[return-value]

        # Forward pass: collect the value at each ancestor level.
        # The traversal is inherently dynamically typed — we walk through
        # dataclass attrs and mapping items whose types aren't known
        # statically.  Any is confined to these locals; the public API
        # uses object for parameters and _R for the return type.
        #
        # We stop one step short of the leaf: the backward pass at step i
        # uses intermediates[i] as the parent to reconstruct, so we need
        # intermediates[0..n-1] (the root through the second-to-last
        # step's value).  The leaf value itself is never read — we're
        # replacing it — so the key may not exist yet (insert case).
        current: Any = self._root
        intermediates: list[Any] = [current]
        for kind, key in self._path[:-1]:
            if kind == _ATTR:
                current = getattr(current, key)  # type: ignore[arg-type]  # key is str for attr steps
            else:
                current = current[key]
            intermediates.append(current)

        # Backward pass: reconstruct frozen copies from leaf to root.
        result: Any = value
        for i in range(len(self._path) - 1, -1, -1):
            kind, key = self._path[i]
            parent = intermediates[i]
            if kind == _ATTR:
                result = dataclasses.replace(parent, **{key: result})  # type: ignore[arg-type]  # key is str for attr steps
            else:
                result = type(parent)({**parent, key: result})

        return result  # type: ignore[return-value]
