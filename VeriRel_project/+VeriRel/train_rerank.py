import jsonlines
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import sigmoid
import os
import shutil
import argparse
from tqdm import tqdm

def set_seed(seed=27):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--train_path', type=str, help='Path to the train file')
parser.add_argument('--dev_path', type=str, help='Path to the validation file')
parser.add_argument('--corpus_path', type=str, help='Path to the corpus file')
parser.add_argument('--path_to_save', type=str, help='Path to the model saving file')
args = parser.parse_args()

set_seed(27)

prf = list(jsonlines.open(args.train_path))
dev = list(jsonlines.open(args.dev_path))
corpus = list(jsonlines.open(args.corpus_path))

max_saved_models = 3
saved_models = []

def save_best_model(epoch, dev_score, path_to_save, model, tokenizer):
    global saved_models
    save_path = os.path.join(path_to_save, f'epoch-{epoch}-loss-{int(dev_score * 1e4)}')
    os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    saved_models.append((save_path, dev_score))
    saved_models = sorted(saved_models, key=lambda x: x[1])
    if len(saved_models) > max_saved_models:
        worst_model_path = saved_models.pop()
        shutil.rmtree(worst_model_path[0])
        print(f"Deleted model at {worst_model_path[0]}")

class RerankDataset(Dataset):
    def __init__(self, claims, corpus):
        self.dataset = []
        candidates_dict = {
            candidate['doc_id']: candidate['title'] + ' ' + ' '.join(candidate['abstract'])
            for candidate in corpus
        }
        for claim in claims:
            claim_text = claim['claim']
            for doc_id, score in zip(claim['doc_ids'], claim['prf_scores']):
                candidate_text = candidates_dict.get(doc_id)
                if candidate_text:
                    self.dataset.append({
                        'query': claim_text,
                        'candidate': candidate_text,
                        'label': score
                    })

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def encode_data(dataset):
    text_pairs = list(zip(dataset['query'], dataset['candidate']))
    encode_text = tokenizer.batch_encode_plus(
        text_pairs,
        max_length=512,
        truncation=True,
        truncation_strategy='only_first',
        padding=True,
        return_tensors='pt')
    encode_text = {key: tensor.to(device) for key, tensor in encode_text.items()}
    return encode_text

def evaluate_model(model, dataset):
    model.eval()
    truths = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size_gpu):
            encoded_text = encode_data(batch)
            logits_result = model(**encoded_text).logits
            probabilities = sigmoid(logits_result)
            truths.extend(batch['label'].float().tolist())
            outputs.extend(probabilities.view(-1).tolist())
    differences = [abs(a - b) for a, b in zip(truths, outputs)]
    return sum(differences) / len(differences)

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', force_download=False)
model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=1)

trainset = RerankDataset(prf, corpus)
dev = RerankDataset(dev, corpus)

epochs = 40
batch_size_gpu = 8
batch_size_accumulated = 64
lr = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 40)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('it is using:  ')
print(device)
print('')
_ = model.to(device)

best_dev_loss = float('inf')
epochs_without_improve = 0
patience = 3

for e in range(epochs):
    print('epoch: ' + str(e))
    model.train()
    t = tqdm(DataLoader(trainset, batch_size=batch_size_gpu, shuffle=True))
    torch.set_grad_enabled(True)
    for i, batch in enumerate(t):
        encoded_text = encode_data(batch)
        output = model(**encoded_text)
        output_logits = output.logits
        sigmoid_logits = sigmoid(output_logits) 
        prf_labels = batch['label'].float().to(device).view(-1, 1)
        loss = torch.mean(torch.abs(sigmoid_logits - prf_labels))
        loss.backward()   
        if (i + 1) % (batch_size_accumulated // batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
    if (i + 1) % (batch_size_accumulated // batch_size_gpu) != 0:
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    train_score = evaluate_model(model, trainset)
    print('train loss:' + str(train_score))
    dev_score = evaluate_model(model, dev)
    print('dev loss:' + str(dev_score))
    save_path = os.path.join(args.path_to_save, f'epoch-{e}-loss-{int(dev_score * 1e4)}')
    save_best_model(e, dev_score, args.path_to_save, model, tokenizer)
    if dev_score + 1e-6 < best_dev_loss:
        best_dev_loss = dev_score
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1
        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break



