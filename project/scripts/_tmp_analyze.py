"""Quickly analyse the recleaned jsonl: print top-worst samples + stats."""
import json
import statistics
import sys

path = sys.argv[1]
samples = []
with open(path) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("type") == "eval":
            samples.append((rec["data"]["cer%"], rec["data"]["pred"],
                            rec["data"]["ref"], rec["id"]))
samples.sort(reverse=True)
print("--- Top 30 worst samples ---")
for s in samples[:30]:
    print(f"id={s[3]:4d} cer={s[0]:6.2f}%  pred={s[1][:60]!r}  ref={s[2][:30]!r}")

total = len(samples)
n100 = sum(1 for s in samples if s[0] >= 99.999)
n50 = sum(1 for s in samples if 50 < s[0] < 99.999)
nlow = sum(1 for s in samples if s[0] < 50)
print()
print("--- Stats ---")
print(f"total={total}  100%={n100}  50<cer<100={n50}  <50%={nlow}")
ss = [s[0] for s in samples]
print(f"mean={statistics.mean(ss):.2f}  median={statistics.median(ss):.2f}")
non_full = [s[0] for s in samples if s[0] < 99.999]
if non_full:
    print(f"mean(excluding 100% items)={statistics.mean(non_full):.2f}")
