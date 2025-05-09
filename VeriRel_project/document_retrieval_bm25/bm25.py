import jsonlines
import heapq
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


class BM25RetrieverWithScores:
    def __init__(self, corpus_path, stop_words=None):
        self.corpus_path = corpus_path
        self.stop_words = stop_words if stop_words else set(stopwords.words('english'))
        self.corpus = list(jsonlines.open(self.corpus_path))
        self.doc_texts = [doc['title'] + ' ' + ' '.join(doc['abstract']) for doc in self.corpus]
        self.tokenized_docs = [self.tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.doc_ids = [doc['doc_id'] for doc in self.corpus]

    def tokenize(self, text):
        return [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in self.stop_words]

    def retrieve(self, query, top_k):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
        top_doc_ids = [self.doc_ids[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        return top_doc_ids, top_scores


def retrieve_all_with_scores(dataset_path, retriever, output_path, top_k):
    dataset = list(jsonlines.open(dataset_path))
    with jsonlines.open(output_path, 'w') as writer:
        for entry in dataset:
            doc_ids, scores = retriever.retrieve(entry['claim'], top_k=top_k)
            writer.write({
                'id': entry['id'],
                'claim': entry['claim'],
                'doc_ids': doc_ids,
                'scores': scores
            })


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    corpus_file = 'dataset/scifact_open/corpus.jsonl'
    dataset_file = 'dataset/scifact_open/claims.jsonl'
    output_file = 'document_retrieval_result/bm25/bm25_scifact_open_500_with_scores.jsonl'
    top_k = 500

    retriever = BM25RetrieverWithScores(corpus_path=corpus_file)
    retrieve_all_with_scores(dataset_file, retriever, output_file, top_k)

