import argparse

from src.crc_py.models.svm_subcategory import SubcategoryClassifier

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out_svm_mv")
    ap.add_argument("--comment", required=True)
    ap.add_argument("--code", default="")
    ap.add_argument("--file_path", default="")
    ap.add_argument("--owner", default="")
    ap.add_argument("--repo", default="")
    args = ap.parse_args()
    clf = SubcategoryClassifier(args.model_dir)
    sub, score = clf.infer(args.comment, code=args.code, file_path=args.file_path, owner=args.owner, repo=args.repo)
    print({"subcategory": sub, "score": round(score, 3)})
