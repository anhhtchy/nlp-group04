import nltk
from pyvi import ViTokenizer
import numpy as np
from gensim.models import KeyedVectors
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Punkt Sentence Tokenizer
print("Downloading punkt...")
try:
    nltk.data.find('tokenizers/punkt')
    print('Punkt Existed')
except LookupError:
    nltk.download('punkt')
    print("Downloaded Punkt")

# Pretrained Word2Vec
vocab = None
print("Loading Word2Vec vocab...")
try:
    vocab_file = open("word2vec-vocab.pkl", "rb")
    vocab = pickle.load(vocab_file)
    vocab_file.close()
except Exception as e:
    print(e)
    w2v = KeyedVectors.load_word2vec_format('we_knn/wiki.vi.vec')
    vocab = w2v.key_to_index
    vocab_file = open("word2vec-vocab.pkl", "wb")
    pickle.dump(vocab, vocab_file)
    vocab_file.close()
print("Done")

def word2vec_summarizer(paragraph:str, n_clusters:int=4):
    sentences = nltk.sent_tokenize(paragraph)
    X = []
    for sentence in sentences:
        sentence = ViTokenizer.tokenize(sentence) # Output "Xin_chào ! Rất vui được gặp bạn ."
        words = sentence.split(" ")
        sentence_vec = np.zeros((300))
        for word in words:
            if word in vocab:
                sentence_vec += vocab[word]
                break # why break
        X.append(sentence_vec)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    return ''.join(summary)