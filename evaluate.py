import glob
import nltk
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
from summarizer_score import bert_score_compute, rouge_score_compute
from pyvi import ViTokenizer
from bertsummarizer import Summarizer
import numpy as np

from word2vec_summarizer import word2vec_summarizer

DATA_DIR = './data/plaintext/'
MANUAL_DIR = './data/manual_summary/'

print("Downloading punkt...")
try:
    nltk.data.find('tokenizers/punkt')
    print('Existed')
except LookupError:
    nltk.download('punkt')
    print("Downloaded")


def get_data():
    data_dir_list = []
    for dir in glob.glob('./data/plaintext/*/*.txt'):
        data_dir_list.append(dir[17:])
    return data_dir_list


def read_original(dir: str):
    dir = DATA_DIR + dir
    file = open(dir, 'r', encoding='utf8')
    original = file.read()
    original = process(original)
    file.close()
    return original


def read_ref(dir: str):
    dir = MANUAL_DIR + dir
    file = open(dir, 'r', encoding='utf8')
    ref = file.read()
    ref = process(ref)
    file.close()
    return ref


def process(para: str):
    processed = ''
    for line in para.splitlines():
        line = line.strip()
        if line != '':
            if line[-1] != '.':
                line = line + '. '
            else:
                line = line + ' '
        processed += line
    return processed.strip()

def evaluate(summarizer_method, metric, rouge_type='2'):
    scores = []
    data_dir_list = get_data()
    count_error = 0
    if summarizer_method == "bert":
        print("Initialize BERT Summarizer...")
        model = Summarizer()
        print("Done")
    for dir in data_dir_list:
        print(dir)
        try:
            ref = read_ref(dir)
            original = read_original(dir)
        except:
            count_error += 1
            continue
        ref_sentences = nltk.sent_tokenize(ref)
        n_clusters = len(ref_sentences)
        print(n_clusters)
        if n_clusters < 2:
            continue
        summary = ""
        
        if summarizer_method == "bert":
            try:
                summary = ''.join(model(
                    body=original,
                    ratio=float(n_clusters),
                    min_length=30,
                    use_first=False
                ))
                summary = summary.replace('_', ' ')
                if metric == "bert":
                    p_, r_, f1_ = bert_score_compute(summary, ref, 'vi')
                    scores.append([p_, r_, f1_])
                if metric == "rouge":
                    p, r, f1 = rouge_score_compute(summary, ref, rouge_type)
                    scores.append([p, r, f1])
            except AssertionError:
                pass
            except ValueError:
                pass
        
        if summarizer_method == "w2v":
            summary = word2vec_summarizer(original, n_clusters)

            if summary != "":
                if metric == "bert":
                    try:
                        p_, r_, f1_ = bert_score_compute(summary, ref, 'vi')
                        scores.append([p_, r_, f1_])
                        # print(bert_w2v)
                    except AssertionError:
                        pass
                if metric == "rouge":
                    p, r, f1 = rouge_score_compute(summary, ref, rouge_type)
                    scores.append([p, r, f1])
    print("count error: ", count_error)
    return scores

if __name__ == '__main__':
    # f = open("evaluate-result.txt")
    # f.close()

    # rouge_w2v = np.array(evaluate("w2v", "rouge"))
    # rouge_bert = np.array(evaluate("bert", "rouge"))
    bert_w2v = np.array(evaluate("w2v", "bert"))
    # rouge_w2v = np.array(rouge_w2v)
    print("p    r   f1")
    # print(np.mean(rouge_w2v, axis=0))
    # print(np.mean(rouge_bert, axis=0))
    print(np.mean(bert_w2v, axis=0))
    # print(np.mean(rouge_w2v, axis=0))
