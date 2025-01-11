import mlflow
import re

import json

import pandas as pd 
from markdown import markdown
from bs4 import BeautifulSoup

from transformers import BertTokenizer, BertModel
import torch


import warnings
warnings.filterwarnings("ignore")


THRESHOLD = 0.4
SEED = 123

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
bert_model = BertModel.from_pretrained("prajjwal1/bert-small")

def _clean_markdown(text):
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["img", "a"]):
        tag.decompose()

    clean_text = soup.get_text()
    clean_text = re.sub(r'[*_~`]', '', clean_text)
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def _get_bert_embeddings(texts, max_len=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    return outputs[1].numpy()


def _build_dataframe(data):
    ds = pd.DataFrame(data["issues"])
    ds["title"] = ds["title"].str.lower()
    ds["body"] = ds["body"].str.lower()
    ds["body"].fillna("", inplace=True)
    ds["clean_body"] = ds["body"].apply(_clean_markdown)

    return ds


def _get_embedding_vectors(ds):
    texts = list(ds["title"] + " " + ds["clean_body"])
    return _get_bert_embeddings(texts)


def load_model(mlflow_server_uri: str, model_run_id: str, model_name: str, model_version: int):
    mlflow.set_tracking_uri(uri=mlflow_server_uri)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_run_id,
        artifact_path="config.json",
    )

    with open(local_path, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.keras.load_model(model_uri)

    return model_config, model


def run_query(model, model_config, issue):
    ds = _build_dataframe({ "issues": [ issue ] })
    embedding_vectors = _get_embedding_vectors(ds)
    label_classes = model_config["project_labels"]

    probabilities = model.predict(embedding_vectors, verbose=0)
    predicted_classes_flags = (probabilities > THRESHOLD).astype(int)[0]
    predicted_classes = [ label_classes[class_index] for class_index, is_predicted in enumerate(predicted_classes_flags) if is_predicted ]

    return predicted_classes
