# -*- coding: utf-8 -*-
"""
ngram_experiment.py —— 人民日报 n-gram 语言模型实验（作业 1 主实验）

实验流程（对应 PPT 第 11 页编程作业）：
    1. 语料清洗    ：去掉行号 ID、{注音}、全角数字归一化（为什么要清洗？见 README）
    2. 建模训练    ：从零实现 n-gram 计数 + 6 种概率估计方法，对比训练时间
       - mle     最大似然（不平滑，作为“反面教材”基线）
       - laplace 加一平滑（add-1 / Lidstone 特例）
       - jm      Jelinek-Mercer 线性插值（λ 用 EM 在验证集上估计）
       - katz    Katz 回退（Good-Turing 折扣，n_r 缺失时退化为绝对折扣）
       - kn      插值 Kneser-Ney（续接计数 + 单一折扣 D = n1/(n1+2n2)）
       - mkn     改进 Kneser-Ney（D1/D2/D3+ 分段折扣）
    3. 评测        ：留出集困惑度（PPL）对比
    4. 文本生成    ：以给定开头续写，比较不同 n、不同平滑、贪婪 vs 温度采样
    5. （可选）用 NLTK 的 nltk.lm 在小子集上对照，验证自实现 KN 的正确性

运行示例：
    python ngram_experiment.py                          # 默认：200 万词，全方法，2/3/4-gram
    python ngram_experiment.py --fast-demo              # 快速演示（约 1 分钟）
    python ngram_experiment.py --full                   # 全部 730 万词（需约 3-4GB 内存）
"""

import argparse
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache

from rmrb_corpus import (BOS, EOS, UNK, load_clean_sentences, tokenize_prompt)

INF = math.inf
log = print  # 简单起见直接 print；报告另写入 results/report.md


# ================================================================
# 第 2 步：n-gram 计数（“训练”的本体）
# ================================================================

class NGramCounter:
    """一次扫描建立 1..n 阶全部原始计数。

    self.counts[j]   : dict ctx(tuple, 长度 j) -> Counter(word -> count)
                       即 (j+1)-gram 的计数表；j = 0 时 ctx = () 为 unigram。
    self.totals[j]   : dict ctx -> c(ctx) = sum_w c(ctx, w)，随计数同步累加，
                       避免以后每次求上下文总频都要对 Counter 求和（大分支如
                       “的”后面可能跟几千种词，重复求和会非常慢）。
    """

    def __init__(self, order: int):
        self.order = order
        self.counts = [defaultdict(Counter) for _ in range(order)]
        self.totals = [defaultdict(int) for _ in range(order)]

    def add_sentence(self, toks):
        """计入一个已加哨兵的句子：[BOS]*(n-1) + words + [EOS]"""
        n = self.order
        for i in range(n - 1, len(toks)):
            w = toks[i]
            for k in range(1, n + 1):          # 同时累计 k = 1..n 阶计数
                ctx = tuple(toks[i - k + 1:i])
                self.counts[k - 1][ctx][w] += 1
                self.totals[k - 1][ctx] += 1


def build_counts(sentences, order, report_every=5000):
    """计数构建 + 记录各阶耗时（用于“对比训练时间”）。

    返回 (counter, per_order_seconds, total_words)。
    注意：加哨兵后每个位置都要为 k=1..n 各计一次，所以总操作数 ≈ 词数 × n，
    观察输出里的耗时增长即可理解“训练成本随 n 线性增长，而模型表大小随 n 指数膨胀”。
    """
    C = NGramCounter(order)
    t_start = time.perf_counter()
    t_k = [0.0] * order
    n_words = 0
    for si, s in enumerate(sentences):
        toks = [BOS] * (order - 1) + s + [EOS]
        n_words += len(toks) - (order - 1)
        for i in range(order - 1, len(toks)):
            w = toks[i]
            for k in range(1, order + 1):
                t0 = time.perf_counter()
                ctx = tuple(toks[i - k + 1:i])
                C.counts[k - 1][ctx][w] += 1
                C.totals[k - 1][ctx] += 1
                t_k[k - 1] += time.perf_counter() - t0
        if (si + 1) % report_every == 0:
            ngram_types = sum(len(d) for d in C.counts)
            log(f"  [计数] 已处理 {si + 1} 句 / 累计 n-gram 类型数 {ngram_types:,} / "
                f"用时 {time.perf_counter() - t_start:.1f}s")
    return C, t_k, n_words


# ================================================================
# 第 3 步：n-gram 语言模型（6 种概率估计方法）
# ================================================================

class NGramLM:
    """统一接口：logprob(ctx, w, method)。

    ctx 是长度恰为 n-1 的 tuple（句子开头由 <s> 补齐）。
    各平滑方法的公式与实现动机见各方法内注释与 README。
    """

    KMX = 6  # Katz / Good-Turing 折扣的上限计数阈值

    def __init__(self, order, counter, vocab):
        self.n = order
        self.C = counter
        self.vocab = vocab            # list，含 <s> </s> <unk>
        self.vocab_set = set(vocab)
        self.V = len(vocab)
        self._jm_lambdas = None
        self._gt_all = None           # {上下文长度 m: GT 折扣表}，惰性构建
        self._kn = None               # 续接计数结构，惰性构建
        self._kn_disc = None          # 每阶折扣，惰性构建

    # ---------------------------------------------------------- 计数工具
    def _cnt(self, ctx, w):
        """c(ctx, w)；ctx 长度 m（0..n-1），查 (m+1)-gram 表"""
        succ = self.C.counts[len(ctx)].get(ctx)
        return succ.get(w, 0) if succ else 0

    def _tot(self, ctx):
        return self.C.totals[len(ctx)].get(ctx, 0)

    # ---------------------------------------------------------- MLE / Laplace
    def _p_mle(self, ctx, w):
        tot = self._tot(ctx)
        return (self._cnt(ctx, w) / tot) if tot else 0.0

    def _p_laplace(self, ctx, w):
        """add-1（Laplace）：把每个词条先“白送”1 次计数，词表越小越极端。
        V 大时概率被严重压平，PPL 会高得离谱——这正是它的教学意义。"""
        return (self._cnt(ctx, w) + 1.0) / (self._tot(ctx) + self.V)

    # ---------------------------------------------------------- Jelinek-Mercer
    def fit_jm(self, val_sents, iters=20):
        """用 EM（deleted interpolation）在验证集上估计插值权重 λ。

        p_JM(w|ctx) = Σ_{j=1..n} λ_j · p_MLE^{(j)}(w|ctx 的最近 j-1 个词)
        EM 迭代：
            λ_j^{new} = (1/T) Σ_t  [ λ_j p_j(w_t|h_t) / Σ_k λ_k p_k(w_t|h_t) ]
        初始化为均匀分布，通常 10~20 轮收敛。
        """
        n = self.n
        lam = [1.0 / n] * n
        log(f"  [JM] 在 {sum(len(s) for s in val_sents):,} 词验证集上 EM 估计 λ（{iters} 轮）……")
        for it in range(iters):
            numer = [0.0] * n
            total = 0
            for s in val_sents:
                toks = [BOS] * (n - 1) + s + [EOS]
                for i in range(n - 1, len(toks)):
                    w = toks[i]
                    full_ctx = tuple(toks[i - n + 1:i])
                    # j 从 1 到 n：分量阶数 n..1，上下文长度取 n-j（j=1 全量，j=n 为空）
                    comps = [self._p_mle(full_ctx[-(n - j):] if n - j > 0 else (), w)
                             for j in range(1, n + 1)]
                    denom = sum(lam[j - 1] * comps[j - 1] for j in range(1, n + 1))
                    if denom <= 0:
                        continue
                    for j in range(1, n + 1):
                        numer[j - 1] += lam[j - 1] * comps[j - 1] / denom
                    total += 1
            if total:
                lam = [x / total for x in numer]
        self._jm_lambdas = lam
        log(f"  [JM] λ = {[round(x, 4) for x in lam]}（高阶在前）")

    def _p_jm(self, ctx, w):
        lam = self._jm_lambdas
        p = 0.0
        for j in range(1, self.n + 1):          # 分量阶数 n..1，上下文长度取 n-j
            p += lam[j - 1] * self._p_mle(ctx[-(self.n - j):] if self.n - j > 0 else (), w)
        return p

    # ---------------------------------------------------------- Katz 回退
    def _gt_discounts_all(self):
        """为每个上下文长度 m（1..n-1）各算一套 Good-Turing 折扣。

        c*_r = (r+1)·n_{r+1}/n_r（r < KMX），r ≥ KMX 不打折；若某些 r 的
        n_r = 0（数据不够大时常见），GT 公式失效，该阶退化为绝对折扣
        D = n1/(n1+2n2)。注意：Katz 必须在每一阶都做折扣——否则低阶把
        概率质量全部锁死在已见词上（α=0），未见续接只能拿到 0 概率。
        """
        out = {}
        for m in range(1, self.n):
            fr = Counter()                       # fr[r] = 该阶计数为 r 的 n-gram 类型数
            for succ in self.C.counts[m].values():
                for c in succ.values():
                    if c <= self.KMX:
                        fr[c] += 1
            if any(fr[r] == 0 for r in range(1, self.KMX)):
                n1, n2 = fr[1], fr[2]
                D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5
                out[m] = {r: max(r - D, 0.0) for r in range(1, self.KMX + 1)}
            else:
                disc = {}
                for r in range(1, self.KMX):
                    disc[r] = min(r, (r + 1) * fr[r + 1] / fr[r])
                disc[self.KMX] = self.KMX
                out[m] = disc
        return out

    @lru_cache(maxsize=100_000)
    def _katz_alpha(self, ctx):
        """α(ctx) = 该上下文扣下的总质量 / 低阶模型未覆盖的质量。

        无续接词的上下文返回 None 表示“整体回退”。
        两个比值的分子分母都可能接近 0，用差量求和与 log1p 计算，
        避免 (1 - x) 在 x→1 时的灾难性消去误差。
        """
        m = len(ctx)
        succ = self.C.counts[m].get(ctx)
        if not succ:
            return None
        tot = self._tot(ctx)
        leftover = sum(max(cc - self._gt_all[m].get(cc, cc), 0.0)
                       for cc in succ.values()) / tot
        low_mass = sum(self._katz(ctx[1:], w2) for w2 in succ)
        if leftover <= 0.0 or low_mass >= 1.0:
            return 0.0
        return math.exp(math.log(leftover) - math.log1p(-low_mass))

    @lru_cache(maxsize=500_000)
    def _katz(self, ctx, w):
        """Katz 回退：见过的高阶 n-gram 用折扣计数；没见过的把概率质量
        按 α(ctx) 回退给低阶分布（每一阶都有折扣）。"""
        m = len(ctx)
        if m == 0:
            return self._p_mle((), w)                      # unigram MLE 兜底
        c = self._cnt(ctx, w)
        tot = self._tot(ctx)
        if tot == 0:                                       # 训练中从未见过的上下文
            return self._katz(ctx[1:], w)                  # 整体回退
        if c > 0:
            return self._gt_all[m].get(c, c) / tot         # 本阶折扣计数
        alpha = self._katz_alpha(ctx)
        if alpha is None:
            return self._katz(ctx[1:], w)
        return alpha * self._katz(ctx[1:], w)

    def _p_katz(self, ctx, w):
        if self._gt_all is None:
            self._gt_all = self._gt_discounts_all()
            log(f"  [Katz] 各阶 GT 折扣表（阶 m: 计数1/2/3 的折扣）: "
                f"{ {m: [round(d[r], 3) for r in (1, 2, 3)] for m, d in self._gt_all.items()} }")
        return self._katz(ctx, w)

    # ---------------------------------------------------------- Kneser-Ney 家族
    def _build_kn(self):
        """构建各低阶的“续接计数”（continuation counts）：

            c_cont(ctx, w) = |{ w0 : c(w0, ctx, w) > 0 }|
        即 (ctx, w) 前面出现过多少种不同的词。思想见 README 思考题 1。
        同时统计续接总频，并预计算每阶折扣（避免每次查询都重扫整张表）。
        """
        n = self.n
        cont = {m: defaultdict(Counter) for m in range(1, n)}   # m=1..n-1
        cont_tot = {m: defaultdict(int) for m in range(1, n)}
        for m in range(1, n):                                   # 由 (m+1)-gram 表推导 m 阶续接计数
            for ctx_full, succ in self.C.counts[m].items():     # ctx_full 长度 m
                ctx = ctx_full[1:]                              # 长度 m-1
                for w in succ:
                    cont[m][ctx][w] += 1                        # 每个 (m+1)-gram 类型贡献 1
                cont_tot[m][ctx] += len(succ)                   # 续接总频 = 类型数之和
        self._kn = (cont, cont_tot)

        # 每阶折扣：m == n 用原始计数，m < n 用续接计数（KN 与普通插值的本质区别）。
        # 返回 (单一 D, (D1, D2, D3)) 供插值 KN 与改进 KN 分别取用。
        self._kn_disc = {}
        for m in range(2, n + 1):
            cnt = Counter()                                     # 计数分布 n1, n2, ...
            if m == n:
                for succ in self.C.counts[m - 1].values():
                    for c in succ.values():
                        cnt[c] += 1
            else:
                for succ in cont[m].values():
                    for c in succ.values():
                        cnt[c] += 1
            n1, n2 = cnt[1], cnt[2]
            D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5
            Y = D
            n3, n4 = cnt[3], cnt[4]
            D1 = 1 - 2 * Y * n2 / n1 if n1 else D
            D2 = 2 - 3 * Y * n3 / n2 if n2 else D
            D3 = 3 - 4 * Y * n4 / n3 if n3 else D
            self._kn_disc[m] = (D, (max(D1, 0.0), max(D2, 0.0), max(D3, 0.0)))

    @lru_cache(maxsize=100_000)
    def _succ_stats(self, m, ctx):
        """训练表中 ctx 后续词的统计：(N1+, N1, N2, N3+)。

        最高阶查原始计数表，低阶查续接计数表；用于计算插值项 γ。
        """
        if m == self.n:
            succ = self.C.counts[m - 1].get(ctx)
        else:
            succ = self._kn[0][m].get(ctx)
        if not succ:
            return (0, 0, 0, 0)
        n1 = n2 = n3 = 0
        for c in succ.values():
            if c == 1:
                n1 += 1
            elif c == 2:
                n2 += 1
            elif c >= 3:
                n3 += 1
        return (len(succ), n1, n2, n3)

    @lru_cache(maxsize=500_000)
    def _kn_rec(self, m, ctx, w, modified):
        """KN 插值递归：

            p_kn(w|ctx) = max(c(ctx,w)-D, 0)/c(ctx)
                          + γ(ctx) · p_kn(w|ctx 的最近 m-2 个词)
        最高阶用原始计数，其余阶全部用续接计数；m=1 时回到续接 unigram。
        归一化项 γ 必须与折扣配套，保证概率质量守恒：
            普通 KN ：γ = D·N1+(ctx·)/c(ctx)
            改进 KN ：γ = [D1·N1 + D2·N2 + D3+·N3+]/c(ctx)
        注意：评测/生成期间一律用 .get 查询，避免给 defaultdict 插入新键
        （否则每问一个未见上下文都会让表越查越大，内存失控）。
        """
        n = self.n
        if m == 1:                                          # 续接 unigram 兜底
            tot = self._kn[1][1].get((), 0)                 # 全部 bigram 类型数
            base = self._kn[0][1].get(())
            c = base.get(w, 0) if base else 0
            return c / tot if tot else 0.0
        D, D3 = self._kn_disc[m]
        if m == n:                                          # 最高阶：原始计数
            c = self._cnt(ctx, w)
            tot = self._tot(ctx)
            if tot == 0:
                return self._kn_rec(m - 1, ctx[1:], w, modified)
        else:                                               # 低阶：续接计数
            succ = self._kn[0][m].get(ctx)
            c = succ.get(w, 0) if succ else 0
            tot = self._kn[1][m].get(ctx, 0)
            if tot == 0:
                return self._kn_rec(m - 1, ctx[1:], w, modified)
        # 折扣 d：改进 KN 按 c 分档，普通 KN 统一用 D
        if c >= 1:
            d = D3[0] if (modified and c == 1) else \
                D3[1] if (modified and c == 2) else \
                D3[2] if (modified and c >= 3) else D
        else:
            d = 0.0
        # 插值项 γ：与折扣方案配套（普通 KN 只有 D·N1+ 被扣下；MKN 按三档分计）
        _, n1, n2, n3 = self._succ_stats(m, ctx)
        if modified:
            gamma = (D3[0] * n1 + D3[1] * n2 + D3[2] * n3) / tot
        else:
            gamma = D * (n1 + n2 + n3) / tot
        return max(c - d, 0.0) / tot + gamma * self._kn_rec(m - 1, ctx[1:], w, modified)

    def _p_kn(self, ctx, w, modified=False):
        if self._kn is None:
            self._kn_build_time_start = time.perf_counter()
            self._build_kn()
            log(f"  [{'M' if modified else ''}KN] 续接计数构建完成，"
                f"用时 {time.perf_counter() - self._kn_build_time_start:.1f}s")
        return self._kn_rec(self.n, ctx, w, modified)

    # ---------------------------------------------------------- 统一入口
    def logprob(self, ctx, w, method):
        """返回 log p(w|ctx)。MLE 未见过时返回 -inf（零概率问题正是要展示的）。"""
        if method == "mle":
            p = self._p_mle(ctx, w)
            return math.log(p) if p > 0 else -INF
        if method == "laplace":
            return math.log(self._p_laplace(ctx, w))
        if method == "jm":
            return math.log(max(self._p_jm(ctx, w), 1e-12))
        if method == "katz":
            return math.log(max(self._p_katz(ctx, w), 1e-12))
        if method == "kn":
            return math.log(max(self._p_kn(ctx, w), 1e-12))
        if method == "mkn":
            return math.log(max(self._p_kn(ctx, w, modified=True), 1e-12))
        raise ValueError(method)

    # ---------------------------------------------------------- 评测与生成
    def perplexity(self, sents, method):
        """留出集困惑度 PPL = exp(-平均对数似然)。返回 (ppl, 词数, 秒)。"""
        t0 = time.perf_counter()
        total_logp = 0.0
        n_words = 0
        for s in sents:
            toks = [BOS] * (self.n - 1) + s + [EOS]
            for i in range(self.n - 1, len(toks)):
                total_logp += self.logprob(tuple(toks[i - self.n + 1:i]), toks[i], method)
                n_words += 1
        ppl = INF if total_logp == -INF else math.exp(-total_logp / n_words)
        return ppl, n_words, time.perf_counter() - t0

    def check_normalized(self, ctx_words, method, label=""):
        """抽查归一化：Σ_{w∈词表} p(w|ctx) 应 ≈ 1（MLE 因零概率不足 1 属正常）。
        这是验证回退/插值实现正确性的重要自检。"""
        ctx = tuple(ctx_words[-(self.n - 1):]) if self.n > 1 else ()
        s = 0.0
        for w in self.vocab:
            p = math.exp(self.logprob(ctx, w, method)) if method != "mle" \
                else self._p_mle(ctx, w)
            s += p
        log(f"     [归一化抽查] ctx={''.join(ctx_words[-3:])}  Σp(w|ctx) = {s:.4f}  ({label})")
        return s

    def generate(self, seed_tokens, method, max_new=40, temperature=None,
                 topk=0, rng_seed=42):
        """从 seed 开头续写文本。

        temperature=None/0 → 贪婪（每步取概率最大词）；
        temperature>0      → 温度采样（topk>0 时只在概率前 k 大的词里采样）。
        """
        rng = random.Random(rng_seed)
        # 开头对齐词表：OOV 词逐字退化，再不行用 <unk>
        seed = []
        for w in seed_tokens:
            if w in self.vocab_set:
                seed.append(w)
            elif w not in (BOS, EOS):
                for ch in w:
                    seed.append(ch if ch in self.vocab_set else UNK)
        ctx = tuple([BOS] * (self.n - 1) + seed)[-(self.n - 1):]
        # 候选词：句首符不能出现在句中；<unk> 计数集中、贪婪时容易自锁成循环，也排除
        cands = [w for w in self.vocab if w not in (BOS, UNK)]
        out = []
        for _ in range(max_new):
            if not temperature:                          # 贪婪
                w = max(cands, key=lambda x: self.logprob(ctx, x, method))
            else:
                logps = [self.logprob(ctx, x, method) for x in cands]
                if topk > 0:                             # top-k 截断
                    idx = sorted(range(len(cands)), key=lambda i: logps[i], reverse=True)[:topk]
                    cands_k = [cands[i] for i in idx]
                    logps = [logps[i] for i in idx]
                else:
                    cands_k = cands
                m = max(logps)
                if m == -INF:
                    # MLE 下所有候选概率都是 0（未见上下文）→ 没有概率可言，均匀随机
                    w = rng.choice(cands_k)
                else:
                    ws = [math.exp((lp - m) / temperature) for lp in logps]
                    w = rng.choices(cands_k, weights=ws)[0]
            if w == EOS:
                break
            out.append(w)
            ctx = ctx[1:] + (w,)
        return out


# ================================================================
# 第 1 步补充：数据划分（训练 / 验证 / 测试）与词表裁剪
# ================================================================

def split_data(sents, limit, eval_tokens, jm_val_tokens):
    """截取语料并划分：训练 / JM 验证 / 留出测试。

    全部按句累计词数，达到 limit 即止（limit=None 表示用全部）；
    测试集从尾部取，JM 验证集取训练集尾部，三者互不重叠、保持原文顺序
    （近似时间切分，评测更诚实）。
    """
    picked, n_tok = [], 0
    for s in sents:
        n_tok += len(s)
        picked.append(s)
        if limit is not None and n_tok >= limit:
            break
    if len(picked) < 100:
        raise SystemExit("语料太少，请增大 --limit")
    # 尾部切测试集
    test, t = [], 0
    while picked and t < eval_tokens:
        s = picked.pop()
        test.append(s)
        t += len(s)
    # 训练集尾部切 JM 验证集
    val, v = [], 0
    while picked and v < jm_val_tokens:
        s = picked.pop()
        val.append(s)
        v += len(s)
    train = picked
    log(f"[划分] 训练 {sum(len(s) for s in train):,} 词 / JM验证 {v:,} 词 / "
        f"测试 {t:,} 词（共 {len(train)}/{len(val)}/{len(test)} 句）")
    return train, val, test


def build_vocab(train_sents, min_count):
    """按训练集词频建词表：低频词统一替换为 <unk>（稀疏问题的标准缓解）。"""
    freq = Counter(w for s in train_sents for w in s)
    keep = {w for w, c in freq.items() if c >= min_count}
    vocab = sorted(keep) + [UNK, BOS, EOS]
    n_unk = sum(1 for s in train_sents for w in s if w not in keep)
    log(f"[词表] min_count={min_count} → 词表大小 {len(vocab):,}；"
        f"训练集低频词被替换为 <unk> 共 {n_unk:,} 次")
    return vocab, keep


def apply_vocab(sents, keep):
    """把不在 keep 集合里的词替换为 <unk>（测试集必须同样处理）。"""
    return [[w if w in keep else UNK for w in s] for s in sents]


# ================================================================
# 第 5 步：NLTK nltk.lm 对照实验（小子集，验证自实现正确性）
# ================================================================

def nltk_comparison(train, test, order, our_result, our_discount, out_lines):
    """在同一个小子集上分别跑 nltk.lm 与我们的实现，对比 PPL。

    若二者（同为插值 KN）PPL 接近，说明自实现正确。
    nltk 的 KN 只支持固定折扣，传入我们的最高阶 D 使口径尽量一致
    （剩余差异来自我们低阶按各自 n1/n2 计 D、nltk 全阶统一 D）。
    """
    try:
        from nltk.lm import MLE, Laplace, KneserNeyInterpolated
        from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
        from nltk.util import everygrams
    except ImportError:
        log("  [NLTK] 未安装 nltk，跳过对照（pip install nltk 后可补做）")
        return
    log(f"  [NLTK] 在 {sum(len(s) for s in train):,} 词子集上训练对照模型（order={order}）……")
    # 注意：padded_everygram_pipeline 返回的是生成器（外层和内层 everygrams 都是），
    # 多个模型复用前必须把两层都物化成 list，否则第一个模型会把流消费完，
    # 后面的模型等于在空语料上训练。
    train_padded, vocab = padded_everygram_pipeline(order, train)
    train_padded = [list(g) for g in train_padded]
    vocab = list(vocab)
    if UNK not in vocab:
        vocab.append(UNK)                          # 保证测试集 OOV 有处可去
    vocab_set = set(vocab)
    test_padded = [list(pad_both_ends([w if w in vocab_set else UNK for w in s], n=order))
                   for s in test]
    # nltk 的 perplexity 接受的是扁平的 everygram 元组流；只用满阶 n-gram
    # （min_len=max_len=order），与我们的评测口径一致：每个位置都评
    # P(w_i | w_{i-n+1}^{i-1})，混入低阶会把口径搞乱。
    test_ngrams = [eg for s in test_padded for eg in everygrams(s, min_len=order, max_len=order)]
    rows = []
    for name, cls, kw in [("MLE", MLE, {}), ("Laplace", Laplace, {}),
                          ("KneserNeyInterpolated", KneserNeyInterpolated, {"discount": our_discount})]:
        t0 = time.perf_counter()
        m = cls(order, **kw)
        m.fit(train_padded, vocabulary_text=vocab)
        fit_t = time.perf_counter() - t0
        ppl = m.perplexity(test_ngrams)
        log(f"     nltk {name:>22}: 训练 {fit_t:6.1f}s  PPL = {ppl:10.2f}")
        rows.append((name, fit_t, ppl))
    out_lines += [
        "\n## 5. NLTK 对照（验证自实现）\n",
        f"- 子集规模：训练 {sum(len(s) for s in train):,} 词，测试 {sum(len(s) for s in test):,} 词，order = {order}\n",
        "| 模型 | 训练时间(s) | PPL |",
        "| --- | ---: | ---: |",
    ]
    for name, fit_t, ppl in rows:
        out_lines.append(f"| nltk {name} | {fit_t:.1f} | {ppl:,.2f} |")
    if our_result is not None:
        our_ppl = our_result
        out_lines.append(f"| 自实现 kn | — | {our_ppl:,.2f} |")
        out_lines.append("\n> 两行插值 KN 的 PPL 应基本一致（都在同一子集上评测），"
                         "这是对自实现正确性的直接检验。\n")
    return rows


# ================================================================
# 主流程
# ================================================================

def run(args):
    out = []          # 报告正文（markdown 行）
    t_all = time.perf_counter()

    # ---------- 1. 清洗与加载 ----------
    log("=" * 64)
    log("【1/5】语料清洗")
    cache = os.path.join(args.outdir, "cleaned_corpus.txt")
    sents = load_clean_sentences(args.data_dir, cache_file=cache)
    out.append("# 人民日报 n-gram 语言模型实验报告\n")
    out.append("## 1. 数据与清洗\n")
    out.append(f"- 语料：1998 年 1—6 月《人民日报》切分标注版（“带音”版本），GBK 编码，共 {len(sents):,} 句\n")
    out.append("- 清洗内容：行号 ID（日期-版次-文章-句号）、多音字注音 `{shang5}` 式标注、全角数字归一化。"
               "**为什么必须清洗**：若不清洗，`上{shang5}` 与 `上` 会被当成两个不同的词，计数被分散、"
               "词表虚胖、稀疏加剧；行号 ID 会被当成普通词参与 n-gram 计数，污染上下文。\n")

    # ---------- 2. 划分与词表 ----------
    log("=" * 64)
    log("【2/5】数据划分与词表裁剪")
    train, val, test = split_data(sents, args.limit, args.eval_tokens, args.jm_val_tokens)
    vocab, keep = build_vocab(train, args.min_count)
    train = apply_vocab(train, keep)
    val = apply_vocab(val, keep)
    test = apply_vocab(test, keep)

    # ---------- 3. 逐阶计数（训练） ----------
    log("=" * 64)
    log("【3/5】n-gram 计数训练（对比训练时间）")
    counters, count_times, model_results = {}, {}, {}
    out.append("\n## 2. 训练时间对比\n")
    out.append("| n | 计数用时(s) | n-gram 类型数 |")
    out.append("| ---: | ---: | ---: |")
    for order in args.orders:
        log(f"-- 构建 {order}-gram 计数表……")
        C, t_k, n_words = build_counts(train, order)
        ngram_types = sum(len(d) for d in C.counts)
        counters[order] = C
        count_times[order] = (t_k, sum(t_k), ngram_types)
        out.append(f"| {order} | {sum(t_k):,.1f} | {ngram_types:,} |")
        log(f"   累计 {ngram_types:,} 个 n-gram 类型，各阶耗时(秒) "
            f"{ {k+1: round(t,1) for k, t in enumerate(t_k)} }")
    out.append("\n> 观察点：n 每 +1，类型数通常膨胀一个量级（近似 |V|^n 的指数增长），"
               "而计数耗时只线性增长——**存储与稀疏才是 n-gram 的真正瓶颈**。\n")

    # ---------- 4. 各方法 PPL 评测 ----------
    log("=" * 64)
    log("【4/5】困惑度（PPL）评测")
    out.append("\n## 3. 困惑度对比（留出测试集）\n")
    out.append("| 方法 \\ n | " + " | ".join(str(n) for n in args.orders) + " | 说明 |")
    out.append("| --- | " + " | ".join("---:" for _ in args.orders) + " | --- |")
    notes = {
        "mle": "不平滑：未见过即 0 概率",
        "laplace": "add-1：概率被压平",
        "jm": "JM 插值，λ 由 EM 估计",
        "katz": "Katz 回退 + GT 折扣",
        "kn": "插值 Kneser-Ney",
        "mkn": "改进 Kneser-Ney（D1/D2/D3）",
    }
    results = {}   # (order, method) -> ppl
    for order in args.orders:
        C = counters[order]
        for method in args.methods:
            model = NGramLM(order, C, vocab)
            t0 = time.perf_counter()
            if method == "jm":
                model.fit_jm(val)
            elif method in ("katz", "kn", "mkn"):
                ppl, nw, ev_t = model.perplexity(test, method)   # 首次调用会触发结构构建
                setup = time.perf_counter() - t0 - ev_t
                log(f"  n={order} {method:>8}: PPL = {ppl:,.2f}  (评测 {ev_t:.1f}s, "
                    f"结构构建 {setup:.1f}s)")
                results[(order, method)] = ppl
                model_results[(order, method)] = model
                continue
            ppl, nw, ev_t = model.perplexity(test, method)
            log(f"  n={order} {method:>8}: PPL = {ppl:,.2f}  (评测 {ev_t:.1f}s)")
            results[(order, method)] = ppl
            model_results[(order, method)] = model
    for method in args.methods:
        row = [f"{'∞' if math.isinf(results[(o, method)]) else f'{results[(o, method)]:,.2f}'}"
               for o in args.orders]
        out.append(f"| {method} | " + " | ".join(row) + f" | {notes[method]} |")
    out.append("\n> 观察点：MLE 的 PPL 为 ∞（测试集里必然出现训练未见的组合，概率连乘被 0 拉爆）；"
               "Laplace 因词表巨大（数万）被过度压平，PPL 极高；KN/MKN 通常最好。\n")

    # 归一化抽查（自检实现正确性）
    log("\n  —— 归一化自检 ——")
    probe_ctx = ["在", "我们", "的"]
    for method in [m for m in args.methods if m != "mle"]:
        if (args.orders[-1], method) in model_results:
            model_results[(args.orders[-1], method)].check_normalized(probe_ctx, method, method)

    # ---------- 5. 文本生成 ----------
    log("=" * 64)
    log("【5/5】文本生成：不同 n / 平滑 / 采样策略续写")
    seed_text = args.seed_text
    seed_tokens = tokenize_prompt(seed_text)
    log(f"  开头：「{seed_text}」 → 分词：{' '.join(seed_tokens)}")
    out.append("\n## 4. 文本生成\n")
    out.append(f"- 开头：**{seed_text}**（分词：{' '.join(seed_tokens)}）\n")
    gen_orders = [o for o in args.orders if o in args.gen_orders] or [args.orders[-1]]
    gen_methods = [m for m in args.methods if m in args.gen_methods] or args.methods[:3]
    out.append("- 对比维度：n ∈ {" + ",".join(map(str, gen_orders)) + "} × 平滑 ∈ {" +
               ",".join(gen_methods) + "} × {贪婪, 温度采样}\n")
    out.append("| n | 平滑 | 策略 | 续写结果 |")
    out.append("| ---: | --- | --- | --- |")
    for order in gen_orders:
        for method in gen_methods:
            if (order, method) not in model_results:
                continue
            model = model_results[(order, method)]
            for strat, temp in [("贪婪", None), ("温度0.8", 0.8)]:
                rng_seed = 42
                gen = model.generate(seed_tokens, method, max_new=args.gen_length,
                                     temperature=temp, topk=args.gen_topk, rng_seed=rng_seed)
                text = "".join(gen) if args.join else " ".join(gen)
                log(f"    n={order} {method:>8} {strat}: {seed_text}{text}")
                out.append(f"| {order} | {method} | {strat} | {seed_text}{text} |")
    out.append("\n> 观察点：n 越大，句子越“通顺”（局部搭配越像报纸）；MLE 在未见上下文处"
               "概率全 0，贪婪生成容易陷入重复或提前 </s>；温度越高越“胡言乱语”但多样性越好；"
               "KN 的续写通常比 MLE 更自然——因为回退质量给了“续接多样性高”的词。\n")

    # ---------- NLTK 对照 ----------
    log("=" * 64)
    log("【附】NLTK 对照")
    if not args.no_nltk:
        # 用训练集前 nltk_limit 词的小子集做对照
        sub, t = [], 0
        for s in train:
            sub.append(s)
            t += len(s)
            if t >= args.nltk_limit:
                break
        sub_train, sub_test = sub[:int(len(sub) * 0.9)], sub[int(len(sub) * 0.9):]
        # 用我们自己的实现在同一子集上训练一个 kn 模型对照；
        # 注意评测口径必须与 nltk 侧一致：子测试集中不在子词表里的词先映射为 <unk>
        C_sub, _, _ = build_counts(sub_train, args.nltk_order, report_every=10 ** 9)
        vocab_sub = sorted({w for s in sub_train for w in s}) + [UNK, BOS, EOS]
        sub_test = apply_vocab(sub_test, set(vocab_sub))
        our_model = NGramLM(args.nltk_order, C_sub, vocab_sub)
        our_ppl, _, _ = our_model.perplexity(sub_test, "kn")
        our_D = our_model._kn_disc[args.nltk_order][0]
        log(f"  [对照] 自实现 kn（同子集）: PPL = {our_ppl:,.2f}（最高阶折扣 D = {our_D:.3f}）")
        nltk_comparison(sub_train, sub_test, args.nltk_order, our_ppl, our_D, out)

    # ---------- 收尾 ----------
    out.append("\n## 附：复现命令\n")
    out.append("```\n" + " ".join(sys.argv) + "\n```\n")
    os.makedirs(args.outdir, exist_ok=True)
    report_path = os.path.join(args.outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    log("=" * 64)
    log(f"报告已写入 {report_path}")
    log(f"总用时 {time.perf_counter() - t_all:,.1f}s")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="人民日报 n-gram 语言模型实验")
    ap.add_argument("--data-dir", default="人民日报数据")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--limit", type=int, default=2_000_000,
                    help="使用的语料上限（词数）；0 或 --full 用全部")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fast-demo", action="store_true", help="快速演示：小数据量少组合")
    ap.add_argument("--min-count", type=int, default=2, help="词频低于此值的词替换为 <unk>")
    ap.add_argument("--orders", type=lambda s: [int(x) for x in s.split(",")], default=[2, 3, 4])
    ap.add_argument("--methods", type=lambda s: s.split(","),
                    default=["mle", "laplace", "jm", "katz", "kn", "mkn"])
    ap.add_argument("--eval-tokens", type=int, default=100_000, help="留出测试集词数")
    ap.add_argument("--jm-val-tokens", type=int, default=50_000, help="JM λ 估计验证集词数")
    ap.add_argument("--seed-text", default="在阳光明媚的五月，我们学校胜利召开了")
    ap.add_argument("--gen-length", type=int, default=40)
    ap.add_argument("--gen-topk", type=int, default=0)
    ap.add_argument("--gen-orders", type=lambda s: [int(x) for x in s.split(",")],
                    default=None, help="参与生成的阶数（默认与 --orders 相同）")
    ap.add_argument("--gen-methods", type=lambda s: s.split(","), default=["mle", "laplace", "kn"],
                    help="参与生成的方法（默认 mle,laplace,kn，其余方法只看 PPL）")
    ap.add_argument("--join", action="store_true", help="生成结果不加空格直接拼接")
    ap.add_argument("--nltk-limit", type=int, default=100_000, help="NLTK 对照子集词数")
    ap.add_argument("--nltk-order", type=int, default=3)
    ap.add_argument("--no-nltk", action="store_true")
    args = ap.parse_args(argv)
    if args.full:
        args.limit = 0
    if args.fast_demo:
        args.limit = 300_000
        args.orders = [2, 3]
        args.methods = ["mle", "laplace", "kn"]
        args.eval_tokens = 30_000
        args.jm_val_tokens = 20_000
        args.gen_length = 25
        args.no_nltk = True
    if args.gen_orders is None:
        args.gen_orders = args.orders
    if args.limit == 0:
        args.limit = None
    return args


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())
