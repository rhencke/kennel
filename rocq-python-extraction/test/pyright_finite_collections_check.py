import FiniteCollectionFixtures

fixtures = FiniteCollectionFixtures.FiniteCollectionFixtures

positive_map_check: dict[int, str] = fixtures.positive_task_map
positive_hit_check: str | None = fixtures.positive_task_hit
positive_elements_check: list[tuple[int, str]] = fixtures.positive_task_elements
positive_set_check: frozenset[int] = fixtures.positive_claim_set
positive_set_removed_check: frozenset[int] = fixtures.positive_claim_removed
positive_set_elements_check: list[int] = fixtures.positive_claim_elements

string_map_check: list[tuple[str, int]] = fixtures.string_label_map
string_hit_check: int | None = fixtures.string_label_hit
string_elements_check: list[tuple[str, int]] = fixtures.string_label_elements
string_set_check: list[str] = fixtures.string_label_set(["alpha", "beta"])
string_set_elements_check: list[str] = fixtures.string_label_set_elements(
    ["alpha", "beta"]
)
