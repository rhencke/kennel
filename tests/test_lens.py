"""Tests for fido.lens — path-recording updater for frozen dataclass trees."""

from dataclasses import dataclass

from frozendict import frozendict

from fido.lens import Lens

# ── test fixtures ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Inner:
    value: int


@dataclass(frozen=True)
class Middle:
    items: frozendict[str, Inner]


@dataclass(frozen=True)
class Root:
    middle: Middle
    label: str


def _sample_root() -> Root:
    return Root(
        middle=Middle(
            items=frozendict({"a": Inner(1), "b": Inner(2)}),
        ),
        label="original",
    )


# ── single-level attribute ───────────────────────────────────────────────


class TestLensSingleAttribute:
    def test_set_attribute(self) -> None:
        root = _sample_root()
        result = Lens(root).label.set("changed")
        assert result.label == "changed"
        assert result.middle is root.middle  # untouched subtree is shared

    def test_original_unchanged(self) -> None:
        root = _sample_root()
        Lens(root).label.set("changed")
        assert root.label == "original"


# ── single-level item ───────────────────────────────────────────────────


class TestLensSingleItem:
    def test_set_existing_item(self) -> None:
        root = _sample_root()
        result = Lens(root).middle.items["a"].set(Inner(99))
        assert result.middle.items["a"] == Inner(99)
        assert result.middle.items["b"] is root.middle.items["b"]

    def test_set_new_item(self) -> None:
        root = _sample_root()
        result = Lens(root).middle.items["c"].set(Inner(3))
        assert result.middle.items["c"] == Inner(3)
        assert len(result.middle.items) == 3
        # Existing entries preserved.
        assert result.middle.items["a"] is root.middle.items["a"]


# ── deep navigation ─────────────────────────────────────────────────────


class TestLensDeepNavigation:
    def test_attr_item_attr(self) -> None:
        """Navigate attr → item → attr through the full tree."""
        root = _sample_root()
        result = Lens(root).middle.items["a"].value.set(42)
        assert result.middle.items["a"].value == 42
        assert result.label == "original"
        assert result.middle.items["b"] is root.middle.items["b"]

    def test_preserves_frozendict_type(self) -> None:
        root = _sample_root()
        result = Lens(root).middle.items["a"].set(Inner(10))
        assert type(result.middle.items) is frozendict


# ── empty path ───────────────────────────────────────────────────────────


class TestLensEmptyPath:
    def test_set_at_root(self) -> None:
        root = _sample_root()
        replacement = Root(
            middle=Middle(items=frozendict({})),
            label="replaced",
        )
        result = Lens(root).set(replacement)
        assert result is replacement


# ── FidoState-shaped usage ───────────────────────────────────────────────


@dataclass(frozen=True)
class RepoState:
    key: str
    started_at: str  # simplified for test


@dataclass(frozen=True)
class FidoState:
    repos: frozendict[str, RepoState]


class TestLensFidoStatePattern:
    """Mirrors the actual registry.py update pattern."""

    def test_insert_repo(self) -> None:
        state = FidoState(repos=frozendict())
        new_repo = RepoState(key="owner/repo", started_at="now")
        result = Lens(state).repos["owner/repo"].set(new_repo)
        assert result.repos["owner/repo"] is new_repo

    def test_update_existing_repo(self) -> None:
        old_repo = RepoState(key="owner/repo", started_at="then")
        state = FidoState(repos=frozendict({"owner/repo": old_repo}))
        new_repo = RepoState(key="owner/repo", started_at="now")
        result = Lens(state).repos["owner/repo"].set(new_repo)
        assert result.repos["owner/repo"] is new_repo

    def test_update_preserves_sibling_repos(self) -> None:
        a = RepoState(key="a/a", started_at="t1")
        b = RepoState(key="b/b", started_at="t2")
        state = FidoState(repos=frozendict({"a/a": a, "b/b": b}))
        new_a = RepoState(key="a/a", started_at="t3")
        result = Lens(state).repos["a/a"].set(new_a)
        assert result.repos["a/a"] is new_a
        assert result.repos["b/b"] is b

    def test_with_atomic_reference_update(self) -> None:
        """End-to-end: Lens inside AtomicReference.update()."""
        from fido.atomic import AtomicReference

        ref: AtomicReference[FidoState] = AtomicReference(FidoState(repos=frozendict()))
        new_repo = RepoState(key="owner/repo", started_at="now")
        ref.update(lambda root: Lens(root).repos["owner/repo"].set(new_repo))
        assert ref.get().repos["owner/repo"] is new_repo
