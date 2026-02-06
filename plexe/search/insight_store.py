"""
Insight store for accumulating learnings from search.

Simple CRUD store for insights extracted during model search.
"""

import logging
from datetime import datetime

from plexe.models import Insight

logger = logging.getLogger(__name__)


class InsightStore:
    """
    Simple store for insights extracted from experiments.

    Provides basic CRUD operations for storing and retrieving insights.
    """

    def __init__(self):
        """Initialize empty insight store."""
        self.insights: list[Insight] = []
        self._next_id = 0

    def add(
        self,
        change: str,
        effect: str,
        context: str,
        confidence: str,
        supporting_evidence: list[int],
    ) -> Insight:
        """
        Add new insight.

        Args:
            change: What was varied
            effect: Observed outcome
            context: When this applies
            confidence: "high", "medium", or "low"
            supporting_evidence: Solution IDs

        Returns:
            Created Insight
        """
        insight = Insight(
            id=self._next_id,
            change=change,
            effect=effect,
            context=context,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            timestamp=datetime.now().isoformat(),
        )

        self.insights.append(insight)
        self._next_id += 1

        logger.info(f"Added insight #{insight.id}: {change} → {effect}")
        return insight

    def update(self, insight_id: int, **updates) -> bool:
        """
        Update insight fields.

        Args:
            insight_id: ID of insight to update
            **updates: Fields to update

        Returns:
            True if updated, False if not found
        """
        for insight in self.insights:
            if insight.id == insight_id:
                for key, value in updates.items():
                    if hasattr(insight, key):
                        setattr(insight, key, value)
                    else:
                        logger.warning(f"Insight has no attribute '{key}'")
                logger.info(f"Updated insight #{insight_id}")
                return True

        logger.warning(f"Insight #{insight_id} not found")
        return False

    def get_all(self) -> list[Insight]:
        """Get all insights."""
        return self.insights

    def __len__(self) -> int:
        """Number of insights in store."""
        return len(self.insights)

    def __repr__(self) -> str:
        return f"InsightStore({len(self.insights)} insights)"

    # ============================================
    # Serialization
    # ============================================

    def to_dict(self) -> dict:
        """Serialize InsightStore to dict for checkpointing."""
        return {
            "_next_id": self._next_id,
            "insights": [insight.to_dict() for insight in self.insights],
        }

    @staticmethod
    def from_dict(d: dict) -> "InsightStore":
        """Deserialize InsightStore from checkpoint dict."""
        from plexe.models import Insight

        store = InsightStore()
        store._next_id = d.get("_next_id", 0)
        store.insights = [Insight.from_dict(insight_dict) for insight_dict in d.get("insights", [])]
        return store
