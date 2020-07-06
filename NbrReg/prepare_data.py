#!/usr/bin/env python3

import argparse
import sklearn.preprocessing as skp
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.io import savemat
import numpy as np


# Lex must be sorted
def bin_search(term, lex, lo=0, hi=-1):
    if hi == -1:
        hi = len(lex)

    if lo == hi:
        raise Exception("Bin search failed")

    mid = lo + (hi - lo) // 2
    if lex[mid] == term:
        return mid

    if term < lex[mid]:
        return bin_search(term, lex, lo=lo, hi=mid)

    return bin_search(term, lex, lo=mid + 1, hi=hi)


def split_data(data):
    allsize = data.shape[0]
    trainsize = allsize // 10 * 8
    cvsize = (allsize - trainsize) // 2
    testsize = allsize - trainsize - cvsize

    train_si = 0
    train_ei = trainsize

    cv_si = train_ei
    cv_ei = train_ei + cvsize

    test_si = cv_ei
    test_ei = cv_ei + testsize
    train_data = data[train_si:train_ei]
    cv_data = data[cv_si:cv_ei]
    test_data = data[test_si:test_ei]
    return train_data, cv_data, test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="ng20_self.mat",
                        help="Output file")
    parser.add_argument("-i", "--input", help="Input ng20 docs")
    args = parser.parse_args()
    docs = []
    str_categories = []

    with open(args.input) as ng20s:
        lines = ng20s.readlines()

    try:
        for i, l in enumerate(lines):
            cat, doc = l.rstrip().split("\t", maxsplit=1)
            if doc:
                docs.append(doc.split())
            else:
                raise Exception(f"Empty doc? line: {i + 1}")

            if cat:
                str_categories.append(cat)
            else:
                raise Exception(f"Empty cat? line: {i + 1}")

    except Exception:
        print(f"Line with problem {i + 1}")
        return 1

    categories = skp.LabelBinarizer().fit_transform(str_categories)
    doc_lens = []
    lexicon = set()
    inv_index = defaultdict(list)
    docs_w_counts = []
    for i, rd in enumerate(docs):
        doc_lens.append(len(rd))
        wrd_count = defaultdict(int)
        for w in rd:
            wrd_count[w] += 1
            lexicon.add(w)

        for word, count in wrd_count.items():
            inv_index[word].append((i, count))

        docs_w_counts.append(wrd_count)

    doc_lens = np.array(doc_lens)
    print(f"lex size: {len(lexicon)}")
    lexicon = sorted(list(lexicon))
    lex_len = len(lexicon)
    idfs = []
    print("Computing idfs")
    for w in lexicon:
        df = len(inv_index[w])
        idfs.append(np.log2((lex_len - df + 0.5) / df + 0.5))

    idfs = np.array(idfs)
    print("Computing term freqs")
    freq_row = []
    freq_col = []
    data = []
    for i, dcount in enumerate(docs_w_counts):
        for w, f in dcount.items():
            freq_row.append(i)
            freq_col.append(bin_search(w, lexicon))
            data.append(f)

    term_freqs = csr_matrix((data, (freq_row, freq_col)),
                            shape=(len(doc_lens), lex_len), dtype=float)

    print("Computing BM25 weights")
    k1 = 1.6
    b = 0.75
    bm25_nom = term_freqs.multiply(idfs)
    bm25_nom = bm25_nom.multiply(k1 + 1)
    bm25_denom = k1 * (1 - b + b * doc_lens / np.mean(doc_lens))
    bm25_denom = bm25_denom.reshape((len(doc_lens), 1))
    bm25_denom = term_freqs._add_sparse(bm25_denom)
    np.reciprocal(bm25_denom.data, out=bm25_denom.data)
    bm25 = bm25_nom.multiply(bm25_denom)
    skp.normalize(bm25, copy=False)

    index = np.arange(bm25.shape[0])
    np.random.shuffle(index)
    bm25 = bm25[index]
    categories = categories[index]

    train_docs, cv_docs, test_docs = split_data(bm25)
    train_cats, cv_cats, test_cats = split_data(categories)
    print(f"Train shape: {train_docs.shape}")
    train_scores = cosine_similarity(train_docs)
    train_knn = (train_scores).argsort()[:, :101]
    train_knn = train_knn[:, 1:]
    train_cats = csr_matrix(train_cats)
    cv_cats = csr_matrix(cv_cats)
    test_cats = csr_matrix(test_cats)
    print(f"Saving data to {args.output}")
    savedict = {"train": train_docs, "cv": cv_docs,
                "test": test_docs, "gnd_train": train_cats,
                "gnd_cv": cv_cats, "gnd_test": test_cats,
                "train_knn": train_knn}
    savemat(args.output, mdict=savedict)
    return 0


if __name__ == "__main__":
    exit(main())
