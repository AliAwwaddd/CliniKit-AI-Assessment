"""End-to-end pipeline: load data -> explore -> train -> evaluate -> predict.

Usage:
    python run.py
"""

from no_show.data import load_dataset
from no_show.evaluate import (
    evaluate_model,
    print_comparison,
    save_confusion_matrix,
    save_feature_importance,
    save_roc_curve,
)
from no_show.predict import predict_examples
from no_show.train import save_models, split_train_test, train_all_models


def explore(df) -> None:
    print("== Basic data exploration ==")
    print(f"Rows: {len(df)}")
    print(f"No-show rate: {df['no_show'].mean():.1%}")
    print("\nColumn types:")
    print(df.dtypes.to_string())
    print("\nSummary statistics:")
    print(df.describe(include="all").transpose().to_string())


def main() -> None:
    df = load_dataset()
    explore(df)

    X_train, X_test, y_train, y_test = split_train_test(df)

    print("\n== Training ==")
    pipelines = train_all_models(X_train, y_train)
    save_models(pipelines)
    print(f"Trained and saved: {', '.join(pipelines)}")

    print("\n== Evaluation ==")
    results = [evaluate_model(name, pipeline, X_test, y_test) for name, pipeline in pipelines.items()]
    print_comparison(results)

    for name, pipeline in pipelines.items():
        save_confusion_matrix(name, pipeline, X_test, y_test)
        save_roc_curve(name, pipeline, X_test, y_test)
        save_feature_importance(name, pipeline)
    print("\nPlots written to output/")

    best = max(results, key=lambda r: r["roc_auc"])["name"]
    print(f"\n== Example predictions ({best}, best ROC-AUC) ==")
    print(predict_examples(pipelines[best]).to_string(index=False))


if __name__ == "__main__":
    main()
