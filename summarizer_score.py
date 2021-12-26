import nltk
from bert_score import score
from bert_score import plot_example
from rouge import Rouge


def bert_score_compute(cands, ref, lang):
    cands = nltk.sent_tokenize(cands)
    ref = nltk.sent_tokenize(ref)
    P, R, F1 = score(cands, ref, lang=lang,
                     model_type="bert-base-multilingual-cased", verbose=True)
    return round(float(P.mean()), 2), float(R.mean()), round(float(F1.mean()), 3)


def plot_similarity_matrix(cand, ref, lang):
    plot_example(cand, ref, lang=lang)


def rouge_score_compute(cands, refs, rouge_type):
    rouge = Rouge()
    scores = rouge.get_scores(cands, refs)[0]
    P = scores["rouge-" + rouge_type]['p']
    F1 = scores["rouge-" + rouge_type]['f']
    R = scores["rouge-" + rouge_type]['r']
    return round(P, 3), round(R, 3), round(F1, 3)

