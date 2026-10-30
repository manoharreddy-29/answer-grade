# AI-Based Educational Assistant for Automated Short-Answer Grading

Implements the pipeline from the presentation: input → validate → semantic analysis (sentence embeddings +
cosine similarity) → rubric concept check → hybrid score + explanation + teacher-review flag.

## Run
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py          # web UI
python evaluate.py            # MAE / within-1-mark / review-flag rate on data/sample_labelled.csv
pytest                        # unit tests
```
First run downloads `all-MiniLM-L6-v2`. If `sentence-transformers` is missing/offline, a lexical fallback is
used (the UI tells you) – it cannot recognise true paraphrases.

## Layout
- `grader/preprocess.py` – normalisation, sentence split
- `grader/embedder.py` – SentenceTransformer + fallback
- `grader/scorer.py` – hybrid scoring, concept coverage, review flags
- `data/questions.json` – question, reference, rubric concepts; `data/sample_labelled.csv` – tiny labelled set
- `evaluate.py`, `app.py`, `tests/`

## Team to-do (for the report)
1. Collect 100+ teacher-labelled answers per question; tune `GradingConfig` weights/thresholds on a train split.
2. Report MAE on a held-out split; compare keyword-only vs embedding-only vs hybrid (ablation).
3. Label each feature planned / in progress / implemented, as your slides require.
4. Note: the 10 sample rows are illustrative, not evidence of accuracy.
