import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import pyreadr
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "results" / "claims_multiclass_model.pkl"
DEFAULT_TRAIN = ROOT / "data" / "claims-clean-example.RData"
DEFAULT_TEST = ROOT / "data" / "claims-test.RData"


def parse_html_text(html: object) -> str:
    if pd.isna(html) or html == "":
        return ""
    soup = BeautifulSoup(str(html), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text_blocks = soup.select("h1, h2, h3, h4, h5, h6, p")
    text = (
        " ".join(block.get_text(" ", strip=True) for block in text_blocks)
        if text_blocks
        else soup.get_text(" ", strip=True)
    )
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def read_rdata_frame(path: Path, object_name: str) -> pd.DataFrame:
    objects = pyreadr.read_r(str(path))
    if object_name not in objects:
        raise ValueError(f"{path} does not contain an object named {object_name!r}")
    return objects[object_name]


def make_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("model", LinearSVC(class_weight="balanced")),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multiclass webpage predictions from the fitted SVM."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the fitted model pickle.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        help="Optionally fit and save a model from this labeled RData file.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST,
        help="RData file containing the unlabeled claims_test data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for an RData file containing pred_df.",
    )
    args = parser.parse_args()

    if args.train:
        train = read_rdata_frame(args.train, "claims_clean")
        if "mclass" not in train or "text_clean" not in train:
            raise ValueError("Training data must contain mclass and text_clean columns")
        train_text = train["text_clean"].fillna("").map(
            lambda text: parse_html_text(text) if text else ""
        )
        model = make_model()
        model.fit(train_text, train["mclass"])
        args.model.parent.mkdir(parents=True, exist_ok=True)
        with args.model.open("wb") as model_file:
            pickle.dump(model, model_file)
        print(f"Saved fitted multiclass model to {args.model}")
    else:
        with args.model.open("rb") as model_file:
            model = pickle.load(model_file)

    test = read_rdata_frame(args.test, "claims_test")
    if ".id" not in test or "text_tmp" not in test:
        raise ValueError("Test data must contain .id and text_tmp columns")
    test_text = test["text_tmp"].fillna("").map(parse_html_text)
    pred_df = test[[".id"]].copy()
    pred_df["mclass.pred"] = model.predict(test_text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pyreadr.write_rdata(str(args.output), pred_df, df_name="pred_df")
        print(f"Saved {len(pred_df)} predictions to {args.output}")
    else:
        print(f"Generated {len(pred_df)} multiclass predictions.")
        print(pred_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
