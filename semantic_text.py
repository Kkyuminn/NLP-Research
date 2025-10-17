from datasets import load_dataset
import spacy

def load_sts_dataset():
    """Load the STS Benchmark dataset using Hugging Face Datasets."""
    dataset = load_dataset('stsb_multi_mt', name='en')
    return dataset

def get_spacy_similarity(sentence1, sentence2, nlp):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    return doc1.similarity(doc2)

if __name__ == "__main__":
    data = load_sts_dataset()
    nlp = spacy.load('en_core_web_md')
    # Example: compute similarity for the first pair in the dev set
    s1 = ['Both the mathematices equation are fraudlent']
    s2 = data['dev'][2]['sentence2']
    score = get_spacy_similarity(s1, s2, nlp)
    print(f"Sentence 1: {s1}")
    print(f"Sentence 2: {s2}")
    print(f"spaCy similarity score: {score}")