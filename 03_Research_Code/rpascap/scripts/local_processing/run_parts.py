"""Chunked pNEUMA processing on the user's computer: processes selected archive members one
by one within a time budget, saving one result per file, so that it can resume across calls.
Usage: python3 run_parts.py <zip> <file_list.csv> <parts_dir> <budget_s> [extra member names...]"""
import sys, time, json, zipfile, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import acquire_and_process as A

ZIP, LIST, PARTS, BUDGET = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
EXTRA = sys.argv[5:]
PARTS.mkdir(parents=True, exist_ok=True)


def work(member):
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as zf:
        df, meta = A.process_member(zf, member)
    base = Path(member).name.replace(".csv", "")
    if df is not None:
        df.to_csv(PARTS / f"{base}.cells.csv.gz", index=False, compression="gzip")
    meta["seconds"] = round(time.time() - t0, 1)
    json.dump(meta, open(PARTS / f"{base}.meta.json", "w"), default=float)
    return base, meta["seconds"]


if __name__ == "__main__":
    t_start = time.time()
    want = list(pd.read_csv(LIST)["file"]) + EXTRA
    with zipfile.ZipFile(ZIP) as zf:
        idx = {Path(i.filename).name: (i.filename, i.file_size) for i in zf.infolist() if i.filename.lower().endswith(".csv")}
    todo = [f for f in want if not (PARTS / f"{f[:-4]}.meta.json").exists()]
    todo.sort(key=lambda f: -idx[f][1])                 # largest first
    print(f"remaining {len(todo)} of {len(want)}", flush=True)
    done = 0
    with ProcessPoolExecutor(2) as ex:
        futs = {}
        it = iter(todo)
        def submit():
            f = next(it, None)
            if f is not None and time.time() - t_start < BUDGET:
                futs[ex.submit(work, idx[f][0])] = f
        for _ in range(2):
            submit()
        while futs:
            for fu in as_completed(list(futs)):
                f = futs.pop(fu)
                b, s = fu.result(); done += 1
                print(f"{b} {s}s", flush=True)
                submit()
                break
    left = len([f for f in want if not (PARTS / f"{f[:-4]}.meta.json").exists()])
    print(f"done this call {done}; left {left}; elapsed {time.time()-t_start:.0f}s", flush=True)
