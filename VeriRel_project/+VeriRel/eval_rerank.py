import jsonlines
import heapq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--corpus_path', type=str, help='Path to the corpus file')
parser.add_argument('--claim_path', type=str, help='Path to the claim file')
parser.add_argument('--bm25_path', type=str, help='Path to the BM25 file')
parser.add_argument('--model_path', type=str, help='Path to the trained model')
parser.add_argument('--path_to_save', type=str, help='Path to save reranked results')
parser.add_argument('--rerank_top_k', type=int, default=10, help='Top-k docs to retain after reranking')
args = parser.parse_args()

corpus = list(jsonlines.open(args.corpus_path))
dataset = list(jsonlines.open(args.claim_path))
bm25_result = list(jsonlines.open(args.bm25_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.to(device)
model.eval()

ouput_jsl = jsonlines.open(args.path_to_save, 'w')
id_to_index = {doc['doc_id']: idx for idx, doc in enumerate(corpus)}
index = 0

for data in tqdm(dataset):
    claim = data['claim']
    bm25_selection_index = bm25_result[index]['doc_ids']
    bm25_selection = [corpus[id_to_index[i]]['title'] + ' ' + ' '.join(corpus[id_to_index[i]]['abstract']) for i in bm25_selection_index]
    rerank_scores = []
    for doc_id, bm25_doc in zip(bm25_selection_index, bm25_selection):
        encoded_input_rerank = (claim, bm25_doc)
        with torch.no_grad():
            encoded_inputs = tokenizer([encoded_input_rerank], padding=True, truncation=True, return_tensors='pt', max_length=512)
            encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
            score = model(**encoded_inputs, return_dict=True).logits.view(-1, ).float()
        rerank_score = score.tolist()[0]
        rerank_scores.append((rerank_score, doc_id))
    rerank_scores.sort(reverse=True, key=lambda x: x[0])
    document_id_rerank = [doc_id for _, doc_id in rerank_scores[:args.rerank_top_k]]
    scores_rerank = [scores for scores, _ in rerank_scores[:args.rerank_top_k]]
    claim_id = data['id']
    ouput_jsl.write({
        'id': claim_id,
        'claim': claim,
        'doc_ids': document_id_rerank,
        'scores': scores_rerank
    })
    index += 1

ouput_jsl.close()
