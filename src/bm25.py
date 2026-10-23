"""
依存ライブラリなしで動く BM25（日本語対応: 文字バイグラム）。

形態素解析器（MeCab等）を使わず、
- 漢字・かな・カタカナの連続は 2-gram（1文字だけの連続は 1-gram）
- 英数字の連続は 1語
に分割する。日本語のキーワード検索ベースラインとして標準的な簡易手法。
"""

import math
import re
import unicodedata
from collections import Counter

import numpy as np

_CJK_RUN = re.compile(r"[一-龯ぁ-んァ-ヶー々]+")
_LATIN_RUN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """文字列をBM25用のトークン列にする。"""

    if not text:
        return []

    text = unicodedata.normalize("NFKC", str(text)).lower()

    tokens = []

    for run in _CJK_RUN.findall(text):

        if len(run) == 1:
            tokens.append(run)
            continue

        tokens.extend(
            run[i:i + 2] for i in range(len(run) - 1)
        )

    tokens.extend(_LATIN_RUN.findall(text))

    return tokens


class BM25:
    """Okapi BM25。fit() した文書集合に対して scores() でスコアを返す。"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = 0
        self.doc_freqs = []
        self.doc_len = np.zeros(0)
        self.avg_len = 0.0
        self.df = Counter()

    def fit(self, texts):

        tokenized = [tokenize(t) for t in texts]

        self.n_docs = len(tokenized)
        self.doc_freqs = [Counter(t) for t in tokenized]
        self.doc_len = np.array(
            [len(t) for t in tokenized], dtype=float
        )
        self.avg_len = (
            float(self.doc_len.mean()) if self.n_docs else 0.0
        )

        self.df = Counter()
        for freqs in self.doc_freqs:
            self.df.update(freqs.keys())

        return self

    def _idf(self, term):
        n = self.df.get(term, 0)
        return math.log(
            1.0 + (self.n_docs - n + 0.5) / (n + 0.5)
        )

    def scores(self, query):
        """全文書に対するスコア配列（生のBM25スコア）を返す。"""

        result = np.zeros(self.n_docs)

        if self.n_docs == 0 or self.avg_len == 0:
            return result

        for term in set(tokenize(query)):

            if term not in self.df:
                continue

            idf = self._idf(term)

            for i, freqs in enumerate(self.doc_freqs):

                tf = freqs.get(term, 0)

                if tf == 0:
                    continue

                denom = tf + self.k1 * (
                    1.0 - self.b
                    + self.b * self.doc_len[i] / self.avg_len
                )

                result[i] += idf * tf * (self.k1 + 1.0) / denom

        return result

    def scores_normalized(self, query):
        """最大値で割って 0〜1 にしたスコア。全部0なら全部0。"""

        raw = self.scores(query)
        top = raw.max() if raw.size else 0.0

        return raw / top if top > 0 else raw