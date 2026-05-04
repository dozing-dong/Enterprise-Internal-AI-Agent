import json
from pathlib import Path

cases = json.loads(
    Path("backend/data/data/local_eval/eval_cases.json").read_text(encoding="utf-8")
)
print(f"Total cases: {len(cases)}")
print()
print("First 3 cases:")
for c in cases[:3]:
    print(f"  [{c['case_id']}] [{c['difficulty']}] {c['question']}")
    print(f"    reference: {c['reference']}")
    print(f"    context_ids: {c['reference_context_ids']}")
    print()
cats: dict[str, int] = {}
for c in cases:
    cats[c["category"]] = cats.get(c["category"], 0) + 1
print("Category distribution:", cats)
diff: dict[str, int] = {}
for c in cases:
    diff[c["difficulty"]] = diff.get(c["difficulty"], 0) + 1
print("Difficulty distribution:", diff)
