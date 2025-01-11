import mlflow
import re
import json
import pandas as pd 
from markdown import markdown
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

@dataclass
class Issue:
    title: str
    body: str
    labels: list[str]


def _clean_markdown(text):
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["img", "a"]):
        tag.decompose()

    clean_text = soup.get_text()
    clean_text = re.sub(r'[*_~`]', '', clean_text)
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def _get_bert_embeddings(bert_model, tokenizer, max_len, texts):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    return outputs[1].numpy()


def _build_dataframe(issue: Issue):
    ds = pd.DataFrame({
        "title": [ issue.title ],
        "body": [ issue.body ],
        "labels": [ issue.labels ],
    })

    ds["title"] = ds["title"].str.lower()
    ds["body"] = ds["body"].str.lower()
    ds["body"].fillna("", inplace=True)
    ds["clean_body"] = ds["body"].apply(_clean_markdown)

    return ds


def _get_embedding_vectors(bert_model, tokenizer, max_len, ds):
    texts = list(ds["title"] + " " + ds["clean_body"])
    return _get_bert_embeddings(bert_model, tokenizer, max_len, texts)


class Model:
    def __init__(self, threshold, max_len, tokenizer, bert_model, model, project_labels):
        self._threshold = threshold
        self._max_len = max_len
        self._tokenizer = tokenizer
        self._bert_model = bert_model
        self._model = model
        self._project_labels = project_labels

    def run(self, issue: Issue) -> list[str]:
        ds = _build_dataframe(issue)
        embedding_vectors = _get_embedding_vectors(self._bert_model, self._tokenizer, self._max_len, ds)
        label_classes = self._project_labels

        probabilities = self._model.predict(embedding_vectors, verbose=0)
        predicted_classes_flags = (probabilities > self._threshold).astype(int)[0]
        predicted_classes = [
            label_classes[class_index]
            for class_index, is_predicted in enumerate(predicted_classes_flags)
            if is_predicted
        ]

        return predicted_classes


def load_model(mlflow_server_uri: str, model_run_id: str, model_name: str, model_version: int) -> Model:
    mlflow.set_tracking_uri(uri=mlflow_server_uri)

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
    bert_model = BertModel.from_pretrained("prajjwal1/bert-small")

    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_run_id,
        artifact_path="config.json",
    )
    with open(local_path, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri)

    return Model(
        threshold=model_config["threshold"],
        max_len=128,
        tokenizer=tokenizer,
        bert_model=bert_model,
        model=model,
        project_labels=model_config["project_labels"],
    )
