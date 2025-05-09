import jsonlines
import heapq
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class BM25Retriever:
    def __init__(self, corpus_path, stop_words=None):
        self.corpus_path = corpus_path
        self.stop_words = stop_words if stop_words else set(stopwords.words('english'))
        self.corpus = list(jsonlines.open(self.corpus_path))
        self.doc_texts = [doc['title'] + ' ' + ' '.join(doc['abstract']) for doc in self.corpus]
        self.tokenized_docs = [self.tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.doc_ids = [doc['cord_id'] for doc in self.corpus]

    def tokenize(self, text):
        return [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in self.stop_words]

    def retrieve(self, query, top_k=None):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k = top_k or len(self.doc_ids)
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=scores.__getitem__)
        return [self.doc_ids[i] for i in top_indices]


def retrieve_all(dataset_path, retriever, output_path, top_k=None):
    dataset = list(jsonlines.open(dataset_path))
    with jsonlines.open(output_path, 'w') as writer:
        for entry in dataset:
            claim = entry['claim']
            result_ids = retriever.retrieve(claim, top_k=top_k)
            writer.write({
                'id': entry['id'],
                'claim': claim,
                'cord_ids': result_ids
            })


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    corpus_file = 'dataset/Check_COVID/corpus.jsonl'
    dataset_file = 'dataset/Check_COVID/Check_COVID_all.jsonl'
    output_file = 'document_retrieval_result/bm25/bm25_checkcovid.jsonl'

    retriever = BM25Retriever(corpus_path=corpus_file)
    retrieve_all(dataset_path=dataset_file, retriever=retriever, output_path=output_file)
