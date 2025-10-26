import argparse, json, re
from pathlib import Path
import pandas as pd

def clean_comment(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"```suggestion.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`+", "", text)
    return text.strip()

def write_chunks(df: pd.DataFrame, out_dir: Path, num_chunks: int, prefix: str, digits: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(df)
    base = total // num_chunks
    sizes = [base]*(num_chunks-1)
    sizes.append(total - sum(sizes))
    start = 0
    for i, sz in enumerate(sizes):
        part = df.iloc[start:start+sz]
        path = out_dir / f"{prefix}{str(i+1).zfill(digits)}.json"
        path.write_text(json.dumps(part.to_dict(orient='records'), ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[ok] wrote {path} (n={len(part)})")
        start += sz

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess and rechunk dataset.")
    ap.add_argument("--input", required=True, help="file/dir/glob of JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_chunks", type=int, default=10)
    ap.add_argument("--prefix", default="chunk_")
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Gather files
    p = Path(args.input)
    files = []
    if p.is_dir():
        files = sorted(p.glob('*.json'))
    else:
        import glob as g
        files = [Path(x) for x in g.glob(str(p))]
    rows = []
    for f in files:
        try:
            raw = json.loads(f.read_text(encoding='utf-8'))
            if isinstance(raw, list):
                rows += raw
            elif isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, list):
                        rows += v
                        break
        except Exception as e:
            print(f"[warn] {f}: {e}")
    if not rows:
        raise SystemExit("No rows loaded")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["comment"]).copy()
    df["comment"] = df["comment"].astype(str).apply(clean_comment)
    df = df[df["comment"].str.strip() != ""]
    df = df.drop_duplicates(subset=["code", "comment"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    write_chunks(df, Path(args.out_dir), args.num_chunks, args.prefix, args.digits)
