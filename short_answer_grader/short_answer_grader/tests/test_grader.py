import json
from grader import Rubric, Concept, grade_answer
from grader.embedder import LexicalEmbedder

q = json.load(open("data/questions.json", encoding="utf-8"))["photosynthesis"]
rub = Rubric([Concept(**c) for c in q["concepts"]])
E = LexicalEmbedder()
g = lambda s: grade_answer(q["reference"], s, rub, 10, embedder=E)

def test_correct_high():
    assert g(q["reference"])["marks"] >= 9

def test_paraphrase_covers_concepts():
    r = g("Plants prepare their food by using sunlight, CO₂ and water. This is called photosynthesis.")
    assert "carbon dioxide" in r["covered"] and r["marks"] >= 6

def test_irrelevant_low():
    assert g("The capital of France is Paris.")["marks"] <= 2

def test_blank_flagged():
    r = g("")
    assert r["marks"] == 0 and r["review"]

def test_very_short_flagged():
    assert g("Photosynthesis.")["review"]
