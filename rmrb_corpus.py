# -*- coding: utf-8 -*-
"""
rmrb_corpus.py —— 人民日报 1998 语料的共享工具模块（rmrb = 人民日报拼音缩写）。

语料文件：人民日报数据/1998-MM-2003版-带音.txt（GBK 编码）
每行格式：
    19980101-01-001-001 迈向 充满 希望 的 新 世纪 —— ...
    |--------------|   |----------------------------------|
    行号 ID            已分词的句子（词与词之间用空格分隔）
    日期-版次-文章-句号

“带音”版本还在多音字后面附加了注音，如 上{shang5}、的{zhe4}，
行首还有行号 ID —— 这些都是“多余的标签”，建模前必须清理，原因见 README。

本模块提供：
    clean_line()                    单行清洗
    iter_clean_sentences()          遍历语料，产出清洗后的句子（可附带版次/月份标签）
    load_clean_sentences()          一次性加载（自动缓存到 results/cleaned_corpus.txt）
    tokenize_prompt()               对输入句子分词（jieba，失败则按字切分）
"""

import glob
import os
import re
import sys

# ---------------------------------------------------------------- 常量

# 行号 ID：19980101-01-001-001 （日期8位-版次2位-文章3位-句号3位）
LINE_ID_RE = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+")

# 注音标注：{shang5} {zhe4} {wei4} ... （多音字拼音 + 声调数字）
PINYIN_RE = re.compile(r"\{[a-z]+\d+\}")

# 全角数字 → 半角（语料中混有 １２３ 与 123，归一化后计数不再分散）
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

# 句首/句尾/未登录词哨兵
BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


def clean_line(line: str):
    """清洗一行语料。

    返回 (id_parts, tokens)：
        id_parts = {'date': '19980101', 'page': '01', 'article': '001', 'sentence': '001'}
        tokens   = 清洗后的词列表
    若该行不是有效语料行（空行等），返回 (None, None)。
    """
    line = line.strip()
    m = LINE_ID_RE.match(line)
    if not m:
        return None, None
    id_str = line[: m.end()].strip()
    date, page, article, sent = id_str.split("-")
    body = line[m.end():]
    body = PINYIN_RE.sub("", body)                 # 去掉 {shang5} 之类的注音
    body = body.translate(FULLWIDTH_DIGITS)        # 全角数字转半角
    tokens = [t for t in body.split() if t]        # 按空格切词，丢弃空串
    if not tokens:
        return None, None
    id_parts = {"date": date, "page": page, "article": article, "sentence": sent}
    return id_parts, tokens


def iter_clean_sentences(data_dir: str):
    """遍历 data_dir 下所有 1998-*.txt，逐句产出 (id_parts, tokens)。

    语料为 GBK 编码；个别字节截断用 errors='ignore' 容错。
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到语料文件")
    for path in files:
        with open(path, "rb") as f:
            text = f.read().decode("gbk", errors="ignore")
        for line in text.splitlines():
            id_parts, tokens = clean_line(line)
            if tokens:
                yield id_parts, tokens


def load_clean_sentences(data_dir: str, cache_file: str = None,
                         log=print):
    """加载全部清洗后的句子；若 cache_file 存在则直接读取缓存。

    返回 [tokens, ...]（列表的列表，每句一个词列表）。
    顺带打印清洗统计（去掉了多少行号、多少注音等），便于“观察”清洗过程。
    """
    if cache_file and os.path.exists(cache_file):
        log(f"[清洗] 发现缓存 {cache_file}，直接读取……")
        sents = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sents.append(line.split())
        log(f"[清洗] 缓存加载完成：{len(sents)} 句")
        return sents

    log("[清洗] 逐行清洗语料（去掉行号 ID 与 {注音}）……")
    sents = []
    n_pinyin = n_ids = n_digits = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, "rb") as f:
            text = f.read().decode("gbk", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            m = LINE_ID_RE.match(stripped)
            if not m:
                continue
            n_ids += 1
            body = stripped[m.end():]
            # 统计要在清洗前做：数一数这条原始文本里有多少注音与全角数字
            n_pinyin += len(PINYIN_RE.findall(body))
            n_digits += sum(body.count(c) for c in "０１２３４５６７８９")
            id_parts, tokens = clean_line(stripped)
            if tokens:
                sents.append(tokens)
    log(f"[清洗] 共清洗 {len(sents)} 句；去除行号 ID {n_ids} 个、"
        f"注音标注 {n_pinyin} 个、全角数字 {n_digits} 个")

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            for s in sents:
                f.write(" ".join(s) + "\n")
        log(f"[清洗] 清洗结果已缓存到 {cache_file}")
    return sents


def tokenize_prompt(text: str):
    """把要续写的开头句子切成词。

    优先用 jieba（与人民日报分词风格接近）；未安装时退化为逐字切分。
    后续在 ngram_experiment 中，词表外的词还会继续退化为单字/<unk>。
    """
    try:
        import jieba
        jieba.setLogLevel(60)  # 静默 jieba 的加载日志
        return list(jieba.lcut(text))
    except ImportError:
        # 兜底：汉字逐字、标点独立成词（人民日报语料中单字词很常见）
        return re.findall(r"[一-鿿]|[\w]+|[^\s\w]", text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # 简单自测：打印前 5 句
    from itertools import islice
    for idp, toks in islice(iter_clean_sentences("人民日报数据"), 5):
        print(idp, "->", " ".join(toks))
