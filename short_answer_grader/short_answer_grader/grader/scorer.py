import re
from dataclasses import dataclass, field
from typing import List, Optional
from .embedder import cosine, get_embedder
from .preprocess import normalize, tokens, split_sentences


@dataclass
class Concept:
    name: str
    aliases: List[str] = field(default_factory=list)  # alternative words/phrases (e.g. "co2")


@dataclass
class Rubric:
    concepts: List[Concept]


@dataclass
class GradingConfig:
    w_similarity: float = 0.5      # must be validated on teacher-labelled data
    w_coverage: float = 0.5
    sim_low: float = 0.20          # similarity mapped to 0 below this
    sim_high: float = 0.85         # ... and to 1 above this
    concept_sem_threshold: float = 0.60  # only used with a real semantic model
    min_words: int = 4
    disagreement_gap: float = 0.35
    uncertain_band: tuple = (0.40, 0.60)
    mark_step: float = 0.5


_default_embedder = None


def _embedder():
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = get_embedder()
    return _default_embedder


def _concept_hit(concept: Concept, answer: str, sentences, emb, cfg):
    norm = normalize(answer)
    for term in [concept.name] + list(concept.aliases):
        t = normalize(term)
        if t and re.search(rf"\b{re.escape(t)}\b", norm):
            return True, "keyword"
    if getattr(emb, "semantic", False):
        cv = emb.encode(concept.name)
        for s in sentences:
            if cosine(cv, emb.encode(s)) >= cfg.concept_sem_threshold:
                return True, "semantic"
    return False, ""


def grade_answer(reference: str, student: str, rubric: Rubric, max_marks: float,
                 cfg: Optional[GradingConfig] = None, embedder=None) -> dict:
    cfg = cfg or GradingConfig()
    emb = embedder or _embedder()
    reasons = []

    if not normalize(student):
        return dict(marks=0.0, max_marks=max_marks, similarity=0.0, coverage=0.0, covered=[],
                    missing=[c.name for c in rubric.concepts], review=True,
                    reasons=["Blank answer - confirm with teacher."], model=emb.name)

    sim = max(0.0, cosine(emb.encode(reference), emb.encode(student)))
    sim_n = min(1.0, max(0.0, (sim - cfg.sim_low) / (cfg.sim_high - cfg.sim_low)))

    sentences = split_sentences(student)
    covered, missing = [], []
    for c in rubric.concepts:
        hit, how = _concept_hit(c, student, sentences, emb, cfg)
        (covered if hit else missing).append(c.name)
    coverage = len(covered) / len(rubric.concepts) if rubric.concepts else sim_n

    total_w = cfg.w_similarity + cfg.w_coverage
    combined = (cfg.w_similarity * sim_n + cfg.w_coverage * coverage) / total_w
    step = cfg.mark_step
    marks = round(combined * max_marks / step) * step
    marks = min(max_marks, max(0.0, marks))

    review = False
    if len(tokens(student)) < cfg.min_words:
        review = True; reasons.append("Very short answer - too little context.")
    if abs(sim_n - coverage) > cfg.disagreement_gap:
        review = True; reasons.append("Similarity and concept coverage disagree.")
    lo, hi = cfg.uncertain_band
    if lo <= combined <= hi:
        review = True; reasons.append("Score is in the uncertain middle band.")
    if not getattr(emb, "semantic", False):
        reasons.append("Using lexical fallback model - paraphrases may be under-scored.")

    return dict(marks=marks, max_marks=max_marks, similarity=round(sim, 3), coverage=round(coverage, 3),
                combined=round(combined, 3), covered=covered, missing=missing, review=review,
                reasons=reasons, model=emb.name)
