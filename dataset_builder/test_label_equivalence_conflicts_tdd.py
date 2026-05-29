from collections import defaultdict

from dataset_builder.orchestrator.pipeline import _build_label_equivalence
from dataset_builder.schemas.benchmark_row import BenchmarkRow
from dataset_builder.schemas.interpretation import Interpretation


def _row(domain: str) -> BenchmarkRow:
    return BenchmarkRow(
        review_id=f"{domain}_1",
        group_id=f"{domain}_g1",
        domain=domain,
        domain_family=domain,
        review_text="sample",
        gold_interpretations=[
            Interpretation(
                aspect_raw="price",
                latent_family="price",
                aspect_canonical="price",
                label_type="explicit",
                sentiment="neutral",
                evidence_text="price",
                evidence_span=[0, 5],
                source="gold",
                support_type="gold",
            )
        ],
    )


def test_no_alias_maps_to_multiple_canonicals() -> None:
    eq = _build_label_equivalence([_row("restaurant"), _row("laptop")])
    alias_to_canon = defaultdict(set)
    for canonical, aliases in eq.items():
        for alias in aliases:
            alias_to_canon[alias].add(canonical)
    conflicts = {a: list(v) for a, v in alias_to_canon.items() if len(v) > 1}
    assert not conflicts
