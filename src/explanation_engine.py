"""Phase 6 — Deterministic Explanation Engine.

Generates human-readable explanations for why two cases are similar.
No LLM calls — purely rule-based for reproducibility.
"""
import json
from collections import Counter


def explain_similarity(query_case: dict, result_case: dict,
                       similarity_score: float) -> dict:
    reasons = []
    shared_details = {}

    # 1. IPC section overlap
    q_ipc = set(query_case.get("ipc_sections", []))
    r_ipc = set(result_case.get("ipc_sections", []))
    shared_ipc = q_ipc & r_ipc
    if shared_ipc:
        reasons.append(
            f"Both cases involve IPC sections: {', '.join(sorted(shared_ipc))}"
        )
        shared_details["shared_ipc"] = sorted(shared_ipc)

    # 2. Same court
    q_court = query_case.get("court", "").strip()
    r_court = result_case.get("court", "").strip()
    if q_court and r_court and q_court.lower() == r_court.lower():
        reasons.append(f"Both cases were heard in {q_court}")
        shared_details["same_court"] = q_court

    # 3. Same verdict
    q_verdict = query_case.get("verdict", "")
    r_verdict = result_case.get("verdict", "")
    if q_verdict and r_verdict and q_verdict == r_verdict:
        verdict_display = q_verdict.replace("_", " ").title()
        reasons.append(f"Both cases resulted in: {verdict_display}")
        shared_details["same_verdict"] = q_verdict

    # 4. Same case type
    q_type = query_case.get("case_type", "")
    r_type = result_case.get("case_type", "")
    if q_type and r_type and q_type == r_type:
        reasons.append(f"Both are {q_type} cases")
        shared_details["same_case_type"] = q_type

    # 5. Same crime type
    q_crime = query_case.get("crime_type", "")
    r_crime = result_case.get("crime_type", "")
    if q_crime and r_crime and q_crime.lower() == r_crime.lower():
        reasons.append(f"Both involve {q_crime}")
        shared_details["same_crime_type"] = q_crime

    # 6. Shared evidence types
    q_ev = set(query_case.get("evidence_types", []))
    r_ev = set(result_case.get("evidence_types", []))
    shared_ev = q_ev & r_ev
    if shared_ev:
        reasons.append(
            f"Both reference: {', '.join(sorted(shared_ev))}"
        )
        shared_details["shared_evidence"] = sorted(shared_ev)

    # 7. Entity overlap (persons, organizations)
    q_ents = query_case.get("entities", {})
    r_ents = result_case.get("entities", {})
    for ent_type in ["persons", "organizations"]:
        q_set = set(q_ents.get(ent_type, []))
        r_set = set(r_ents.get(ent_type, []))
        shared = q_set & r_set
        if shared:
            reasons.append(
                f"Shared {ent_type}: {', '.join(sorted(shared))}"
            )
            shared_details[f"shared_{ent_type}"] = sorted(shared)

    # 8. Similarity strength
    if similarity_score >= 0.9:
        strength = "Very High"
    elif similarity_score >= 0.7:
        strength = "High"
    elif similarity_score >= 0.5:
        strength = "Moderate"
    else:
        strength = "Low"

    # Fallback if no specific reasons found
    if not reasons:
        reasons.append(
            "Cases share similar legal language and context "
            f"(semantic similarity: {similarity_score:.2f})"
        )

    return {
        "similarity_score": round(similarity_score, 4),
        "similarity_strength": strength,
        "reasons": reasons,
        "shared_details": shared_details,
        "summary": f"{strength} similarity ({similarity_score:.2f}). "
                   + reasons[0] if reasons else "Semantic match.",
    }


def explain_gap(gap: dict) -> str:
    sections = ", ".join(gap.get("common_ipc_sections", [])[:3]) or "various"
    score_pct = int(gap.get("inconsistency_score", 0) * 100)
    granted = gap.get("bail_granted_count", 0) + gap.get("acquitted_count", 0)
    rejected = gap.get("bail_rejected_count", 0) + gap.get("convicted_count", 0)
    total = gap.get("total_cases", 0)

    return (
        f"Cluster {gap['cluster_id']}: {total} similar cases involving "
        f"IPC {sections} show {score_pct}% verdict inconsistency. "
        f"{granted} favorable vs {rejected} unfavorable outcomes. "
        f"This suggests potential sentencing disparity worth investigating."
    )
