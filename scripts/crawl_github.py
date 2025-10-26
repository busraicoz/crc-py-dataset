import argparse

from src.crc_py.crawling.github_crawler import crawl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base_path", default=None)
    ap.add_argument("--out", default="data/crawled.json")
    args = ap.parse_args()
    crawl(args.owner, args.repo, base_path=args.base_path, out_path=args.out)
