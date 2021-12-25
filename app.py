from flask import Flask, request
from flask import json
from flask.json import jsonify
from flask.templating import render_template
from flask_cors import CORS
import nltk
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
# from revolution_score import bert_score_compute, rouge_score_compute
from pyvi import ViTokenizer
import numpy as np
# from summarizer import Summarizer
from word2vec_summarizer import word2vec_summarizer

app = Flask(__name__)
CORS(app)

'''
DATA_DIR = './data/plaintext/'
MANUAL_DIR = './data/manual_summary/'

print("Initialize Summarizer...")
model = Summarizer()
print("Done")
'''

@app.route('/')
def index():
    return render_template('base.html')

'''
@app.route('/score', methods=['GET'])
def score_get():
    return render_template('score.html')


@app.route('/score', methods=['POST'])
def score_post():
    request_data = request.json
    plaintext_dir = DATA_DIR + str(request_data["plaintext_dir"])
    manual_summary_dir = MANUAL_DIR + str(request_data["plaintext_dir"])
    print(plaintext_dir, manual_summary_dir)
    modeling = str(request_data["model"])
    method = str(request_data["method"])

    file = open(plaintext_dir, 'r', encoding='utf8')
    plaintext = file.read()
    file.close()
    file = open(manual_summary_dir, 'r', encoding='utf8')
    manual_summary = file.read()
    file.close()

    m_s = process(manual_summary)
    processed = process(plaintext)

    sentences = nltk.sent_tokenize(m_s)

    nsum1 = len(sentences)
    print(nsum1, end=' ')
    summary = ""

    if modeling == 'bert':
        summary = ''.join(model(
            body=processed,
            ratio=float(nsum1),
            min_length=0,
            use_first=False
        ))
        summary = summary.replace('_', ' ')
    if modeling == 'word2vec':
        sentences = nltk.sent_tokenize(plaintext)
        X = []
        for sentence in sentences:
            sentence = ViTokenizer.tokenize(sentence)
            words = sentence.split(" ")
            sentence_vec = np.zeros((300))
            for word in words:
                if word in vocab:
                    sentence_vec += vocab[word]
                    break
            X.append(sentence_vec)
        kmeans = KMeans(n_clusters=nsum1)
        kmeans.fit(X)

        avg = []
        for j in range(nsum1):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(nsum1), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    summary = summary.replace('...', '')
    print(len(summary.strip().split('. ')))
    p, r, f1 = 0, 0, 0

    print(m_s.encode("utf-8"))
    print(summary.encode("utf-8"))

    if method == 'bert':
        p, r, f1 = bert_score_compute(summary, manual_summary, lang='vi')
    if method == 'rouge':
        p, r, f1 = rouge_score_compute(summary, manual_summary, 'l')

    resp = {
        "model-summarized": summary,
        "manual-summarized": m_s,
        "paragraph": plaintext,
        "p": p,
        "r": r,
        "f1": f1
    }
    return jsonify(resp)
'''

@app.route('/word2vec', methods=['GET'])
def knn_get():
    return render_template('knn.html')


@app.route('/word2vec', methods=['POST'])
def knn_post():
    data = request.json
    body = process(str(data["body"]))
    # print(body)
    n_clusters = int(data["n_clusters"])

    summary = word2vec_summarizer(body, n_clusters)
    return jsonify({"summarized": summary})

'''
@app.route('/bert', methods=['GET'])
def bert_get():
    return render_template('bert.html')


@app.route('/bert', methods=['POST'])
def bert_post():
    data = request.json
    ratio = float(data["ratio"])
    min_length = int(data["min_length"])
    body = str(data["body"])
    paragraph = "" # paragraph = process(body)
    for line in body.splitlines():
        line = line.strip()
        if line != '' and line[-1:] != '.':
            line = line + '.'
        paragraph += line.strip()

    result = ''.join(model(
        paragraph,
        ratio,
        min_length=min_length
    ))
    result = result.replace('_', ' ')
    resp = {
        "summarized": result
    }
    return jsonify(resp)
'''

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

if __name__ == "__main__":
    app.run(debug=True, threaded=True)