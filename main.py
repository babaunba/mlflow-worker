import model
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-server", default="http://127.0.0.1:8080")
    parser.add_argument("--model-run-id")
    parser.add_argument("--model-name")
    parser.add_argument("--model-version")
    return parser.parse_args()


def main():
    args = parse_args()
    loaded_model = model.load_model(
        mlflow_server_uri=args.mlflow_server,
        model_run_id=args.model_run_id,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    while True:
        issue_title = input(">>> ")
        issue = model.Issue(
            title=issue_title,
            body="",
            labels=[],
        )

        suggested_labels = loaded_model.run(issue)
        print(f"Suggested labels: {suggested_labels}")


if __name__ == "__main__":
    main()
