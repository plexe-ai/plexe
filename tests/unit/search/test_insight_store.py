"""Unit tests for InsightStore."""

from plexe.search.insight_store import InsightStore


def test_insight_store_add_update_serialize_roundtrip():
    """Add/update and serialize/deserialize should preserve insights."""
    store = InsightStore()

    first = store.add(
        change="param x",
        effect="+1",
        context="ctx",
        confidence="low",
        supporting_evidence=[1],
    )
    second = store.add(
        change="param y",
        effect="-1",
        context="ctx",
        confidence="medium",
        supporting_evidence=[2],
    )

    assert first.id == 0
    assert second.id == 1

    updated = store.update(first.id, effect="+2", confidence="high")
    assert updated
    assert store.insights[0].effect == "+2"
    assert store.insights[0].confidence == "high"

    data = store.to_dict()
    restored = InsightStore.from_dict(data)

    assert restored._next_id == store._next_id
    assert len(restored.insights) == 2
    assert restored.insights[0].change == "param x"
    assert restored.insights[1].change == "param y"
