"""Compare recommended marks with teacher marks: MAE, agreement within 1 mark, review-flag rate."""
import json, csv, sys
from grader import Rubric, Concept, grade_answer

qs = json.load(open("data/questions.json", encoding="utf-8"))
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "data/sample_labelled.csv", encoding="utf-8")))
errs, within1, flags = [], 0, 0
print(f"{'id':>3} {'teacher':>7} {'AI':>5} {'flag':>5}")
for r in rows:
    q = qs[r["question_id"]]
    rub = Rubric([Concept(**c) for c in q["concepts"]])
    res = grade_answer(q["reference"], r["answer"], rub, q["max_marks"])
    t = float(r["teacher_marks"]); e = abs(res["marks"] - t)
    errs.append(e); within1 += e <= 1; flags += res["review"]
    print(f"{r['id']:>3} {t:7.1f} {res['marks']:5.1f} {'Y' if res['review'] else '-':>5}")
n = len(rows)
print(f"\nModel: {res['model']}\nMAE: {sum(errs)/n:.2f}\nWithin 1 mark: {within1/n:.0%}\nReview-flag rate: {flags/n:.0%}")
