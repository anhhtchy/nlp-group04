import nltk
from flask import Flask, request
from flask.json import jsonify
from flask.templating import render_template
from flask_cors import CORS

from bertsummarizer import Summarizer
from summarizer_score import bert_score_compute, rouge_score_compute
from word2vec_summarizer import word2vec_summarizer

app = Flask(__name__)
CORS(app)

DATA_DIR = './data/plaintext/'
MANUAL_DIR = './data/manual_summary/'

print("Initialize Summarizer...")
model = Summarizer()
print("Done")


@app.route('/')
def index():
    return render_template('base.html')


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
    original = file.read()
    file.close()

    file = open(manual_summary_dir, 'r', encoding='utf8')
    ref = file.read()
    file.close()

    original = process(original)
    ref = process(ref)

    ref_sentences = nltk.sent_tokenize(ref)
    original_sentences = nltk.sent_tokenize(original)

    ref_len = len(ref_sentences)
    original_len = len(original_sentences)

    summary = ""

    if modeling == 'bert':
        summary = ''.join(model(
            body=original,
            ratio=float(ref_len),
            min_length=0,
            use_first=False
        ))
        summary = summary.replace('_', ' ')
    if modeling == 'word2vec':
        summary = word2vec_summarizer(original, ref_len)

    print(len(summary.strip().split('. ')))
    p, r, f1 = 0, 0, 0

    print(ref.encode("utf-8"))
    print(summary.encode("utf-8"))

    if method == 'bert':
        p, r, f1 = bert_score_compute(summary, ref, lang='vi')
    if method == 'rouge':
        p, r, f1 = rouge_score_compute(summary, ref, 'l')

    resp = {
        "model-summarized": summary,
        "manual-summarized": ref,
        "paragraph": original,
        "p": p,
        "r": r,
        "f1": f1
    }
    return jsonify(resp)


@app.route('/word2vec', methods=['GET'])
def word2vec_get():
    return render_template('word2vec.html')


@app.route('/word2vec', methods=['POST'])
def word2vec_post():
    data = request.json
    body = process(str(data["body"]))

    n_clusters = int(data["n_clusters"])

    summary = word2vec_summarizer(body, n_clusters)
    return jsonify({"summarized": summary})


@app.route('/bert', methods=['GET'])
def bert_get():
    return render_template('bert.html')


@app.route('/bert', methods=['POST'])
def bert_post():
    data = request.json
    ratio = float(data["ratio"])
    min_length = int(data["min_length"])
    body = str(data["body"])
    paragraph = process(body)

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
