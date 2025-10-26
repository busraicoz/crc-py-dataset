import argparse

from src.crc_py.models.svm_subcategory import SubcategoryClassifier

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to labeled JSON")
    ap.add_argument("--model_dir", default="out_svm_mv")
    args = ap.parse_args()
    SubcategoryClassifier(args.model_dir).train(args.train)
