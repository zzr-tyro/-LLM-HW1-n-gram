# -*- coding: utf-8 -*-
"""
fasttext_classify.py —— 可选编程作业：Gensim 表示学习 + FastText 文本分类
（对应 PPT 第 15 页）

任务：
    1. 用 gensim 训练 FastText 词向量（中文语料上子词 n-gram 落在“字”一级）；
    2. 句子表示 = 句中词向量的平均（最简单的“表示学习”组合方式）；
    3. 自选标签做文本分类，分类数量由少到多：
        2 类  ：头版（01版） vs 非头版
        6 类  ：月份（199801..199806）
        12 类 ：版次（01..12 版）
    4. 观察：分类器只拿到“词向量均值”，能区分版面/月份 → 说明词向量确实
       学到了与主题/文体相关的信息（头版多是社论要闻，各版内容各有侧重）。

运行示例：
    python fasttext_classify.py                       # 默认 150 万词，2 类 + 6 类
    python fasttext_classify.py --labels month        # 只跑 6 类
    python fasttext_classify.py --full                # 全部语料（较慢，内存约 2GB+）
"""

import argparse
import glob
import logging
import math
import os
import sys
import time
from collections import Counter

from rmrb_corpus import clean_line

log = print


def load_labeled(data_dir, limit=None):
    """加载带标签的句子。

    标签取自行号 ID：版次 page（01..12）、月份 month（199801..199806）。
    limit 为词数上限（None 表示全部）。
    注意：按 6 个月份文件轮转（round-robin）采样——若按文件顺序截取，
    小 limit 会全部落在 1 月份里，月份标签只剩一个类，分类无从谈起。
    """
    streams = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, "rb") as f:
            text = f.read().decode("gbk", errors="ignore")
        streams.append([(idp, toks) for line in text.splitlines()
                        for idp, toks in [clean_line(line)] if toks])
    n_files = len(streams)
    pos = [0] * n_files
    done = [False] * n_files
    sents, labels = [], []
    n_tok = 0
    while not all(done) and (limit is None or n_tok < limit):
        for i in range(n_files):
            if done[i]:
                continue
            idp, toks = streams[i][pos[i]]
            pos[i] += 1
            if pos[i] >= len(streams[i]):
                done[i] = True
            sents.append(toks)
            labels.append((idp["page"], idp["date"][:6]))
            n_tok += len(toks)
            if limit is not None and n_tok >= limit:
                break
    log(f"[数据] 加载 {len(sents):,} 句，共 {n_tok:,} 词（跨 {n_files} 个月份文件轮转采样）")
    return sents, labels


def make_label_fn(mode):
    """由 mode 返回 句标签映射 与 类别说明。"""
    if mode == "page2":
        return (lambda p, m: "头版" if p == "01" else "非头版"), "2 类：头版 vs 非头版"
    if mode == "month":
        return (lambda p, m: f"{int(m[4:6])}月"), "6 类：月份"
    if mode == "page12":
        return (lambda p, m: f"{p}版"), "12 类：版次"
    raise ValueError(mode)


def sentence_vec(model, toks):
    """句子向量 = 词向量平均（词表外的词跳过；全 OOV 时返回 None）。"""
    vecs = [model.wv[w] for w in toks if w in model.wv]
    if not vecs:
        return None
    return sum(vecs) / len(vecs)


def run(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        force=True)          # gensim 训练进度走 logging
    out = ["# 人民日报 FastText 表示学习 + 文本分类实验\n"]
    t_all = time.perf_counter()

    # ---------- 1. 数据 ----------
    log("=" * 64)
    log("【1/3】加载语料与标签")
    sents, labels = load_labeled(args.data_dir, args.limit)
    out.append("\n## 1. 数据\n")
    out.append(f"- 句子数：{len(sents):,}；标签取自行号 ID（日期-版次-文章-句号）\n")

    # ---------- 2. FastText 词向量 ----------
    log("=" * 64)
    log("【2/3】训练 FastText 词向量")
    from gensim.models import FastText
    t0 = time.perf_counter()
    ft = FastText(vector_size=args.dim, window=5, min_count=args.min_count,
                  sg=1, epochs=args.epochs, workers=args.workers, seed=42)
    ft.build_vocab(sents)
    ft.train(sents, total_examples=len(sents), epochs=ft.epochs)
    train_time = time.perf_counter() - t0
    log(f"  FastText 训练完成：词表 {len(ft.wv):,}，用时 {train_time:.1f}s")
    out.append("\n## 2. 词向量质量抽查\n")
    out.append(f"- vector_size={args.dim}, min_count={args.min_count}, "
               f"epochs={args.epochs}, sg=1；训练用时 {train_time:.1f}s，词表 {len(ft.wv):,}\n")
    for probe in ["改革", "发展", "经济", "北京"]:
        if probe in ft.wv:
            sims = ft.wv.most_similar(probe, topn=5)
            line = "、".join(f"{w}({s:.3f})" for w, s in sims)
            log(f"  「{probe}」近邻：{line}")
            out.append(f"- 「{probe}」近邻：{line}\n")

    # ---------- 3. 分类（由少到多） ----------
    log("=" * 64)
    log("【3/3】文本分类（标签由少到多）")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    out.append("\n## 3. 分类结果\n")
    out.append("| 标签方案 | 类别数 | 训练样本 | 测试样本 | 准确率 |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    for mode in args.labels:
        label_fn, desc = make_label_fn(mode)
        y = [label_fn(p, m) for p, m in labels]
        # 去掉类别样本过少的句子（保证分层切分可用）
        cnt = Counter(y)
        keep_idx = [i for i, c in enumerate(y) if cnt[c] >= 5]
        X_idx, y_sel = [], []
        for i in keep_idx:
            X_idx.append(sents[i])
            y_sel.append(y[i])
        log(f"\n  —— 标签方案：{desc} ——")
        X = []
        keep = []
        for i, toks in enumerate(X_idx):
            v = sentence_vec(ft, toks)
            if v is not None:
                X.append(v)
                keep.append(y_sel[i])
        log(f"    有效样本 {len(X):,}（丢弃全 OOV 句 {len(X_idx) - len(X):,}）")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, keep, test_size=0.2, stratify=keep, random_state=42)
        clf = LogisticRegression(max_iter=500)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        log(f"    LogisticRegression 训练 {time.perf_counter() - t0:.1f}s，"
            f"测试准确率 = {acc:.4f}")
        out.append(f"| {desc} | {len(set(y_sel))} | {len(X_tr):,} | {len(X_te):,} | {acc:.4f} |")
        out.append("\n```\n" + classification_report(y_te, clf.predict(X_te), zero_division=0)
                   + "```\n")

    # ---------- 收尾 ----------
    out.append("\n## 附：复现命令\n")
    out.append("```\n" + " ".join(sys.argv) + "\n```\n")
    os.makedirs(args.outdir, exist_ok=True)
    report_path = os.path.join(args.outdir, "classify_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    log("=" * 64)
    log(f"报告已写入 {report_path}")
    log(f"总用时 {time.perf_counter() - t_all:,.1f}s")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="人民日报 FastText 表示学习 + 文本分类")
    ap.add_argument("--data-dir", default="人民日报数据")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--limit", type=int, default=1_500_000, help="词数上限；--full 用全部")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--labels", type=lambda s: s.split(","),
                    default=["page2", "month"],
                    help="page2(2类) / month(6类) / page12(12类)，由少到多")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.full:
        args.limit = None
    return args


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())
