import json
import streamlit as st
from grader import Rubric, Concept, GradingConfig, grade_answer

st.set_page_config(page_title="AI Short-Answer Grading Assistant", page_icon="📝")
st.title("📝 AI-Based Educational Assistant – Short-Answer Grading")
st.caption("The AI recommends marks. The teacher makes the final decision.")

qs = json.load(open("data/questions.json", encoding="utf-8"))
qid = st.sidebar.selectbox("Load sample question", ["(custom)"] + list(qs))
q = qs.get(qid, {})

with st.sidebar:
    st.subheader("Scoring weights")
    w_sim = st.slider("Semantic similarity weight", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Concept coverage weight = {1 - w_sim:.2f}")

question = st.text_input("Question", q.get("question", ""))
reference = st.text_area("Reference answer", q.get("reference", ""))
max_marks = st.number_input("Maximum marks", 1.0, 100.0, float(q.get("max_marks", 10)))
concept_txt = st.text_area("Rubric concepts (one per line; aliases after ':' separated by commas)",
    "\n".join(f"{c['name']}: {', '.join(c['aliases'])}" for c in q.get("concepts", [])))
student = st.text_area("Student answer")

if st.button("Recommend marks", type="primary"):
    concepts = []
    for line in concept_txt.splitlines():
        if line.strip():
            name, _, al = line.partition(":")
            concepts.append(Concept(name.strip(), [a.strip() for a in al.split(",") if a.strip()]))
    cfg = GradingConfig(w_similarity=w_sim, w_coverage=1 - w_sim)
    r = grade_answer(reference, student, Rubric(concepts), max_marks, cfg)
    st.metric("Recommended marks", f"{r['marks']} / {r['max_marks']:g}")
    c1, c2 = st.columns(2)
    c1.metric("Semantic similarity", r["similarity"]); c2.metric("Concept coverage", r["coverage"])
    st.success("✓ Covered: " + (", ".join(r["covered"]) or "none"))
    st.error("× Missing: " + (", ".join(r["missing"]) or "none"))
    if r["review"]:
        st.warning("⚑ Flagged for teacher review")
    for why in r["reasons"]:
        st.write("•", why)
    st.caption(f"Model: {r['model']}")
    st.number_input("Teacher's final marks", 0.0, float(max_marks), float(r["marks"]), 0.5)
