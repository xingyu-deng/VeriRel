import jsonlines
from tqdm import tqdm
import argparse
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5


def load_jsonl(file_path):
    return list(jsonlines.open(file_path))


def build_doc_lookup(corpus):
    return {doc['doc_id']: idx for idx, doc in enumerate(corpus)}


def rerank_from_bm25_results(corpus, claims, bm25_results, model_name, output_path):
    reranker = MonoT5(pretrained_model_name_or_path=model_name)
    id_to_index = build_doc_lookup(corpus)

    with jsonlines.open(output_path, 'w') as writer:
        for claim_data, bm25_data in tqdm(zip(claims, bm25_results), total=len(claims)):
            claim = claim_data['claim']
            claim_id = claim_data['id']
            retrieved_ids = bm25_data['doc_ids']

            passages = [
                (doc_id, corpus[id_to_index[doc_id]]['title'] + ' ' + ' '.join(corpus[id_to_index[doc_id]]['abstract']))
                for doc_id in retrieved_ids
            ]
            texts = [Text(text, {'docid': doc_id}, 0) for doc_id, text in passages]
            reranked = reranker.rerank(Query(claim), texts)

            doc_ids = [item.metadata['docid'] for item in reranked]
            scores = [item.score for item in reranked]

            writer.write({
                'id': claim_id,
                'claim': claim,
                'doc_ids': doc_ids,
                'scores': scores
            })


def main():
    parser = argparse.ArgumentParser(description="MonoT5 reranking from BM25 results.")
    parser.add_argument('--corpus', required=True, help='Path to the corpus JSONL file.')
    parser.add_argument('--claims', required=True, help='Path to the claims JSONL file.')
    parser.add_argument('--bm25', required=True, help='Path to BM25 JSONL result file.')
    parser.add_argument('--output', required=True, help='Path to write reranked result.')
    parser.add_argument('--model', default='castorini/monot5-3b-msmarco-10k', help='MonoT5 model name.')

    args = parser.parse_args()

    print("Loading files...")
    corpus = load_jsonl(args.corpus)
    claims = load_jsonl(args.claims)
    bm25_results = load_jsonl(args.bm25)

    print("Starting reranking...")
    rerank_from_bm25_results(corpus, claims, bm25_results, args.model, args.output)
    print(f"Saved reranked results to {args.output}")


if __name__ == '__main__':
    main()
