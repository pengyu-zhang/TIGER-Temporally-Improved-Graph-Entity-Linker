import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def convert_text_to_token_ids_and_embeddings(input_file: str, output_file1: str, output_file2: str, output_dir: str, model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open(input_file, 'r', encoding='utf-8') as input_file_handle:
        lines = input_file_handle.readlines()

    token_ids_list = []
    word_to_token_id = {}

    contexts = []

    for line in lines:
        data = json.loads(line)
        if "text" in data:
            context = data["text"]
            contexts.append(" ".join(context))
        elif "context_left" in data and "context_right" in data:
            context = data["context_left"] + data["context_right"]
            contexts.append(" ".join(context))

    encoded = tokenizer.batch_encode_plus(
        contexts,
        add_special_tokens=False,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    token_ids = encoded['input_ids'].tolist()

    for token_id in token_ids:
        token_ids_list.append(token_id)

        for id in token_id:
            word = tokenizer.decode([id], clean_up_tokenization_spaces=True)
            if word not in word_to_token_id:
                word_to_token_id[word] = id

    embeddings_list = []

    for token_id in token_ids:
        input_ids = torch.tensor([token_id]).to(device)

        with torch.no_grad():
            outputs = model(input_ids)

        embeddings = outputs[0].cpu().numpy().astype(np.float32)
        embeddings_list.append(embeddings.squeeze())

    np.save(os.path.join(output_dir, 'embeddings.npy'), np.array(embeddings_list, dtype=np.float32))

    with open(output_file1, 'w', encoding='utf-8') as output_token_ids_file_handle:
        token_ids_str = '\n'.join([' '.join(map(str, token_ids)) for token_ids in token_ids_list])
        output_token_ids_file_handle.write(token_ids_str)

    with open(output_file2, 'w', encoding='utf-8') as output_word_to_token_id_file_handle:
        for word, token_id in word_to_token_id.items():
            output_word_to_token_id_file_handle.write(f'{word} {token_id}\n')


for year in range(2013, 2023):
    # convert_text_to_token_ids_and_embeddings(f'./documents/{year}_train.json',
    #                                         f'./07_knn_relation_small/{year}/{year}_train_token_id.txt',
    #                                         f'./07_knn_relation_small/{year}/{year}_train_token_text.txt',
    #                                         f'/07_knn_relation_small/{year}/',
    #                                         'bert-base-uncased')
    convert_text_to_token_ids_and_embeddings(f'./documents/{year}_train.json',
                                            f'./07_knn_relation_small/{year}/{year}_train_token_id.txt',
                                            f'./07_knn_relation_small/{year}/{year}_train_token_text.txt',
                                            f'/07_knn_relation_small/{year}/',
                                            'bert-base-uncased')
