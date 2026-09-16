---
title: "命题图谱：知识图谱补充网络的研究"
pubDate: 2026-09-17
description: "来自 only-notes 的原始笔记，保留原文内容并做网页格式适配。"
author: "Westwoods"
tags: [知识图谱, RAG, 研究笔记]
draft: false
---

Proposition Graph: A Necessary Way to exploit the Knowledge Reasoning Potential of Large Models
# Introduction
近些年，大语言模型凭借其自然语言生成能力和推理能力在通用以及专业领域知识问答任务上都获得了亮眼的成绩。然而，由于大语言模型的内部知识难以更新以及机器幻觉的存在，大语言模型生成的回答往往包含过时、不可靠的信息。为了解决这样的问题，许多研究都主张维护一个便于更新的知识库充当大语言模型的「第二大脑」。通过实时从知识库中检索相关的支持材料，大语言模型可以生成更加可靠、透明的回答，克服知识密集型问题（knowledge-intensive problems）带来的挑战。

现有的相关研究聚焦于两种知识库形式：非结构化以及结构化。非结构化知识库存储一组文本文档，通过搜索算法来检索出和问题有关的文档作为支持材料，这类思路往往被归纳为 RAG ([Lewis et al., 2020](#^953bea)， [Wu et al,. 2024](#^8f364a))。然而经典的 RAG 不考虑文档之间的逻辑关联，其往往无法精准检索到推理所需的全部信息，并且会引入大量噪声。因此许多研究转而试图通过以知识图谱为代表的结构化数据来实现更好的问答效果。

然而，现有的研究很少考虑知识图谱在知识表示能力上的局限性。知识图谱要求「局部正确性」，因而只能表示静态的事实知识，对正确性需要一定前提的知识无能为力（e.g. 仅当患者出现细菌感染时才能使用抗生素治疗急性支气管炎），如图 1；经典的知识图谱也无法表示事件之间的时序关系（e.g. 特朗普就任美国总统后，新冠疫情在中国爆发）以及逻辑关系（e.g. 孕妇患高血压会提升其患先兆子痫的风险）。面对这样的问题，过去的方法往往会针对每种特定的情况都设计复杂的结构（e.g.时序知识图谱、规则推理系统等）。这往往会耗费大量的精力并且在涉及多种特定知识类型的时候系统的复杂度会大幅提升。

图 1：急性支气管炎（Acute Bronchitis）通常由病毒引起，不应使用抗生素（Antibiotics）治疗；但是当患者被确认出现细菌感染的时候则应当被诊断为细菌引起的急性支气管炎，需要使用抗生素治疗。这个例子揭示了知识图谱在知识表示能力上的局限性。

针对这样的问题，我们提出了命题图谱（Proposition Graph）这一通用结构来补充无法被知识图谱表示的知识。**我们首先注意到了知识图谱的三元组在形式上可以被看作一个命题（proposition）；而知识图谱则等价于一组被假设为相互独立、永真的命题。从认识论范畴来看，几乎所有的知识的正确性都依赖一定条件，这导致假设往往不成立。这是知识图谱推理出错误结果的一大原因**。这一现象在临床推理问题上最为常见。**命题图谱是独立于知识图谱的一种结构化知识，其将知识和条件都表示为命题，显式地表示命题之间的关系，从而可以处理知识复杂的上下文依赖**。命题图谱可以被附加到现有的知识图谱问答推理系统之上，从命题图谱中检索得到的证据可以作为元规则（meta-rule）用于修正（revise）知识图谱证据，提高证据的自洽性和一致性。

> 加粗部分是本篇文章的核心洞见。

我们设计了一整套涵盖通过非结构化文档构建命题图谱、命题图谱证据检索以及协同推理（synergistical reasoning）在内的管线（pipeline）。为了验证其效果，我们利用它基于 MedlinePlus 这一医学知识网站的公开文本数据以及 Procedural HotpotQA 提供的文档构建一个命题图谱。然后我们以 MindMap, Wen et al., 为知识图谱问答管线，在 GenMedGPT-5K 这一临床推理问答数据集以及 Procedural HotpotQA 上进行了实验。实验结果显示，使用命题图谱优化证据的知识图谱问答系统的表现在多个指标上显著好于原知识图谱问答系统。

> 可能多种知识图谱管线以及基线上的实验还需要补充。只用一个 MindMap 做知识图谱管线好像太单调了，不能充分说明命题图谱修正知识图谱证据的价值。

本工作的主要贡献：
1. 提出了命题图谱这一概念，这是一种表示命题之间关系的结构化数据，可以处理知识之间复杂的上下文依赖。命题图谱提供的证据可以用于修正知识图谱证据，提高知识图谱证据的自洽性和一致性；
2. 搭建了一条涵盖从文本中构建命题图谱、命题图谱证据检索以及多证据协同推理在内的管线，并且使用 MedlinePlus 的公开医学文本数据和 Procedural-HotpotQA 文档构建了两个命题图谱数据集；
3. 使用构建的命题图谱和已有的知识图谱管线组成协同推理框架，在 GenMedGPT-5K 和 Procedural-HotpotQA 数据集上进行了实验。结果表明命题图谱可以通过提高证据质量实现比单纯知识图谱管线更好的问答效果，尤其是可以显著提升临床推理问答的安全性水平。
## Related Work
命题图谱对于大模型推理是必不可少的，因为它能够支持归纳

命题图谱本身是归纳的结果，指代消解和归纳。命题图谱在思维中的作用
### RAG based methods for Question Answering

### KG based methods for Question Answering
> 先谈谈常见方案，即一般来说人们如何处理知识图谱依赖的上下文语境。
# Methodology
本文首先提出了命题图谱的概念，并且基于概念设计并且实现了基于文本数据构建命题图谱的管线。在此基础上我们提出了一个命题图谱与知识图谱协同推理框架，该框架可以适配多种多样的知识图谱问答管线，实现 plug and play 式的提升。

命题图谱与知识图谱协同推理框架的架构如图 2 所示。其主要包含四个部分：
1. 知识图谱证据生成（knowledge graph evidence generation）：通过知识图谱生成一组证据 $\mathcal{G}_{q}\leftarrow f(\mathcal{G},q)$；其中 $q$ 代表用户查询，$\mathcal{G}=\langle \mathcal{V},\mathcal{E} \rangle$ 代表知识图谱，$f$ 是知识图谱证据生成管线，管线生成的最终证据是 $\mathcal{G}_{q}=\langle \mathcal{V}_{q},\mathcal{E}_{q} \rangle$；
2. 命题元规则检索（proposition meta-rule retrieval）：可以分为查询抽取（query extraction）、命题匹配（proposition matching）以及命题推理（proposition reasoning）三个步骤，最终会基于用户查询生成一组元规则（meta-rule）；
3. 协同证据优化（synergistical evidence optimization）：利用元规则对知识图谱证据进行知识修正，处理不可靠、不完整的证据，解决证据内部冲突，形成最终使用的证据；
4. 生成答案：提示 LLM 根据优化后的证据生成答案。在证据支持下，我们可以通过修改提示词满足不同任务的侧重点（比如可解释性、透明度、简练）。

图 2：命题图谱与知识图谱协同推理问答管线示意图。知识图谱证据和命题图谱证据的生成会同时进行，最后在协同证据优化阶段被用于生成最终证据。其中，圆形表示知识图谱实体，方框代表命题；橙色方框表示能从查询中获得证实的上下文依赖，蓝色、灰色方框则是在该条件下得以成立以及被否定的命题。这些命题会作为元规则被用于知识图谱证据的协同优化。
## Proposition Graph
在数理逻辑中，命题被定义为可以被判断真假的陈述句，其中许多命题都可以用三元组 $P=\langle s,p,o \rangle$ 表示。通常来说，我们使用顶点、边的集合也即是图 $\mathcal{G}=\langle \mathcal{V},\mathcal{E} \rangle$ 表示知识图谱；但是知识图谱可以被看作一组永真命题的集合 $\mathcal{G}=\{ P^{(i)} \}$。KGQA 可以看作基于查询 $q$，从知识图谱中检索一组三元组的过程：
$$
\begin{cases}
\mathcal{G}_{q}=f(\mathcal{G},q), & \mathcal{G}_{q} \subset \mathcal{G} \\
a=g(\mathcal{G}_{q},q)
\end{cases}
$$
理论上，构成知识图谱的命题 $P_{i}$ 必须是永真的，或者至少需要在所选模型（model）中达到「符合事实」意义上的真。即：
$$
\forall P(P\in \mathcal{G}\to P)
$$
由于人类认识能力的局限性，严格符合知识图谱要求的命题几乎不存在。绝大多数命题都已经或是将要在某些案例下被证伪（be falsified），因此，这些命题的正确性将需要依赖特定的条件：
$$
\forall P((P\in \mathcal{G} \wedge C_{P})\to P)
$$
其中 $C_{P}$ 代表命题 $P$ 为真所需的条件，其也是一个命题，表示 $P$ 所需的上下文依赖。在过去，人们往往会根据现实的情况单独考虑 $C_{P}$ 的问题（e.g. 为三元组附加上时间戳、置信度等）或是将模型划分为更小的部分从而避免考虑 $C_{P}$ 的问题（e.g. 为小领域单独建立知识图谱、多视角本体等）。这些做法都没有系统地考虑命题之间的关系。我们注意到 $C_{P}$ 本质上也是一个命题，因此我们可以从文本数据中提取命题之间的依赖关系，并且将其也组织为图谱 $\mathcal{C}$。这就是**命题图谱**（Proposition Graph）$\mathcal{C}=\{ \langle P_{\text{sub}}^{(i)},R^{(i)},P_{\text{obj}}^{(i)} \rangle \}$。和知识图谱一样，命题图谱也可以表示为顶点和边的集合 $\mathcal{C}=\langle \mathcal{P},\mathcal{R} \rangle$。因此命题图谱满足和知识图谱一样的形式，可以使用图数据库实现高效的存储和检索。
## Constructing Proposition Graph
命题图谱可以使用端到端的方法从文本中抽取。本文直接将文本分割为一组有长度限制的 chunks，并且提示 LLM 从文本中抽取出命题图谱三元组。为了抽取出更有针对性的命题图谱，我们也针对具体场景设定了**命题图谱本体**（ontology）。详见 [Experiments](#Experiments)。
## Synergistical Reasoning with KG & PG
为了利用命题图谱更好地处理命题之间的条件依赖性，改善现有知识图谱问答系统的问答效果，我们提出了知识图谱-命题图谱协同推理框架（KG-PG Synergistical Reasoning Framework）。该框架通过同时从知识图谱和命题图谱中抽取证据，并且利用 LLM 实现知识图谱证据优化，从而实现更好的问答效果。作为一个 plug and play 形式的创新，此框架的知识图谱部分可以被更换为不同的知识图谱证据提取管线。
### KG Evidence Generation
通常来说，基于信息检索的（Information Retrieval，IR）的 KGQA 系统都可以分为两个步骤：生成证据以及生成答案 （如公式 1）。我们的框架主要使用已有 KGQA 系统生成证据的部分，我们将对知识图谱生成的证据进行优化。通过强化合理的证据、去除在当前场景下可信度较低以及无关的证据，框架可以实现更加精准全面的问答。
$$
\mathcal{G}_{q}=\{ G_{q}^{(i)} \}=f(\mathcal{G},q)
$$
$q$ 是 KGQA 的 query；$\mathcal{G}$ 是用于提供证据的知识图谱；$f(\cdot)$ 根据 $q$ 从 $\mathcal{G}$ 中检索出并且最终生成一组证据 $\mathcal{G}_{q}=\{ G_{q}^{(i)} \}=G^{(1)}_{q}, G^{(2)}_{q}, \dots, G^{(n)}_{q}$。许多 Retrieval based 知识图谱问答系统检索出来的证据 $\mathcal{G}_{q}$ 是 $\mathcal{G}$ 的子图，不过得益于 LLM 强大的泛化能力，我们的系统也支持异构甚至非结构化证据。
### PG Meta-Rules Retrieval
PG Meta-Rules Retrieval 部分基于查询提供的信息从命题图谱中检索出一组元规则用于优化知识图谱证据。我们首先需要利用 LLM 的 NLG 能力充分理解 query，然后在理解 query 的基础上用一组命题表示 query 的上下文：
$$
C^{(1)}_{q}, C^{(2)}_{q}, \dots, C^{(n)}_{q} = c(q)
$$
其中，$q$ 是查询 query； $C^{(1)}_{q}, C^{(2)}_{q}, \dots, C^{(n)}_{q}$，是一组在 query 已经给出的前提下为真的命题，表示当前 query 提供的上下文信息；$c(\cdot)$ 是一个抽取器，负责从 query 中抽取出上下文信息。通过命题的形式，我们以一种通用的形式实现了结构化表示上下文依赖。

抽取的命题 $C_{q}^{(i)}$ 需要在命题图谱中匹配与其语义最相似的命题 $P_{q}^{(i)}$，匹配到的命题会作为种子（seed），在命题图谱上进行检索。我们在这里使用的基于 BERT 的 dense 模型编码以及余弦相似度的方法来实现命题文本的语义匹配：
$$
P^{(i)}_{q}=\arg \max _{P\in \mathcal{C}} \frac{\text{Dense}(P)\cdot \text{Dense}(C_{q}^{(i)})}{\begin{Vmatrix}
\text{Dense}(P)
\end{Vmatrix}\cdot \begin{Vmatrix}
\text{Dense}(C_{q}^{(i)})
\end{Vmatrix}}
$$
> 这一步可能有很大的问题，因为这种「不论如何都要选一个出来」的做法可能会为其引入额外的噪声。

接下来我们使用 DFS 的方法，在命题图谱上检索出一组以种子为起点的路径且长度限制为 $k$ 的路径。这一组路径表示命题之间的依赖关系，我们将其作为元规则参与到知识图谱证据优化之中：
$$
M_{q}^{(i)}=\text{DFS}(P_{q}^{(i)},\mathcal{C},k)=\{\langle P_{q}^{(i)},R_{0},P_{0},R_{1},\dots,R_{k},P_{k} \rangle_{j} \}
$$
其中，$M_{q}^{(i)}$ 是第查询 $q$ 从命题图谱 $\mathcal{C}$ 中查询出来的第 $i$ 组元规则，其本身是一组命题图谱中的路径。所有元规则会被汇总起来用于知识图谱证据优化：
$$
\mathcal{M}_{q}=\bigcup_{i=1}^nM_{q}^{(i)},\mathcal{M}_{q} \subset \mathcal{C}
$$
### Synergistical Evidence Optimization
在获得知识图谱证据 $\mathcal{G}_{q}$ 以及命题图谱提供的元规则集合 $\mathcal{M}_{q}$ 之后，我们使用一个基于 LLMs 的融合层（fusion layer）来实现协同证据优化。在融合层中，我们提示 LLM 首先理解命题图谱提供的元规则，并且提供了一组定义好的操作来引导 LLM 结合元规则和对自己的知识来正确地优化知识图谱证据：
1. KEEP & STRENGTHEN: 该证据在当前模型下基本上代表事实，不需要特别的前提，可以直接保留；
2. KEEP but QUALIFY: 该证据所需的前提在当前 query 下已经被满足，保留它的同时强调其满足的前提；
3. REWORD for Clarity:	该证据大体上符合前两种情况，但是表达过于模糊，需要重述；
4. FLAG for Review: 该证据的正确性存在疑点且无法确认其前提是否在当前上下文被满足，需要检查并且标记；
5. DELETE (Omit): 证据本身错误或其需要的上下文依赖不成立，将其从证据中删除。

LLM 可以使用任意次这些操作来优化知识图谱证据。由于 LLM 良好的泛化性能，我们的融合层能够有效地处理知识图谱提供的不同形式的证据（不论是异构 heterogeneous 的还是非结构化的证据）：
$$
\mathcal{G}_{q}^*= G^{(1)*}_{q}, G^{(2)*}_{q}, \dots, G^{(n)*}_{q}  =h(\mathcal{M}_{q},\mathcal{G}_{q})
$$
此外，LLM 将被要求使用自然语言重述优化过的证据。这样做可以有效利用 LLM 的能力来完成指代消解、去重以及处理证据内部矛盾的任务之余大幅压缩生成长度。优化过的证据可以帮助 LLM 给出更加精准、全面的回答，提升问答系统的性能。
### Answering with Optimized Evidence
在证据得到优化之后，我们直接提示 LLM 基于优化后的证据对 query 作出回答。我们可以提示 LLM 在回答之前做进一步推理或结合证据与内部知识作出回答。这样的做法可以避免 LLM 只单纯地重组证据，通过更多 token 的消耗换取更好的效果。
# Experiments
## Experimental Setup
为了验证命题图谱在复杂问题问答上的价值，我们分别在 GenMedGPT-5K 和 Procedural HotpotQA 两个数据集上验证了知识图谱、命题图谱协同推理系统的表现。我们的系统使用 MindMap 作为知识图谱证据生成管线。MindMap 提出了一种高效的知识图谱证据提取方法，可以生成一组编号的知识图谱证据。编号证据会经过融合层优化后用于支持 LLM 问答。

本文为每个测试数据集都构建了完整的命题图谱。对于 GenMedGPT-5K，我们使用 EMCKG 来提供知识图谱证据；而对于 Procedural HotpotQA，我们制作了一个知识图谱 Procedural HotpotQA Knowledge Graph（PHKG）用于问答。为了验证其效果，我们将其和多个 baselines 进行比较。包括无知识辅助组： LLM vanilla（直接回答）、CoT；RAG 组：BM25、Dense；以及 KG 组 GraphRAG、KG-GPT、MindMap。为了为了公平性（fairness）和透明性（transparency），所有基线都使用开源 LLM Qwen 2.5 32B Instruct 作为 backbone。选择 Instruct 模型的原因是相比于 chat 模型 Instruct 模型具有更好的指令依从性，可以减少因为中间结果不满足格式化要求导致的失败样本。细节见 Appendix。

| Dataset             | Domain         | KG    | Question |
| ------------------- | -------------- | ----- | -------- |
| GenMedGPT-5K        | Clinical QA    | EMCKG | 5452     |
| Procedural HotpotQA | Commonsense QA | PHKG  | 256      |
## Clinical QA
GenMedGPT-5K 是一个临床问答数据集，其要求问答系统理解病人的主诉（complains）并且基于主诉分析病人的情况，并且推荐干预手段以及进一步检测方案。和通用问答相比，临床问答往往更看重理解病人的实际情况，需要处理复杂的条件依赖问题。此外，临床问答对回答还具有安全性要求。综合来看临床问答属于复杂、高难度问答任务，可以充分挑战问答系统的推理性能。
### Metrics
我们首先选取了问答任务常用的 BERT-score 作为基础的 metric 来衡量系统的性能。考虑到 BERT-score 等基于 gram 的字符串比对方法主要判定 prediction 是不是 align with the goldens，无法衡量语义以及逻辑上的相似程度，我们还基于 LLM 设计了一种 LLM-score，其通过输出 Correct（完全正确）、Safe（可接受且安全）、Unsafe（不可接受）三类标签来衡量回答的安全性。为了保证透明性和可复现性，LLM-score 也采用开源模型。详细信息以及提示词见 Appendix。
### Results
| Framework   | P      | R      | F1     | Correct | Safe   | Unsafe |
| ----------- | ------ | ------ | ------ | ------- | ------ | ------ |
| Vanilla     | 0.6846 | 0.8028 | 0.7384 | 0.0622  | 0.7854 | 0.1524 |
| CoT         | 0.6840 | 0.7922 | 0.7350 | 0.0552  | 0.7922 | 0.1526 |
| RAG (BM25)  | 0.7458 | 0.8011 | 0.7717 | 0.0798  | 0.7880 | 0.1322 |
| RAG (Dense) |        |        |        |         |        |        |
| KG-GPT      |        |        |        |         |        |        |
| GraphRAG    |        |        |        |         |        |        |
| MindMap     | 0.7223 | 0.8002 | 0.7587 | 0.0551  | 0.4901 | 0.4548 |
| PG+MindMap  | 0.7315 | 0.7977 | 0.7627 | 0.0752  | 0.6191 | 0.3057 |

## Procedural Reasoning
### Metrics
### Results
## Ablation Study
### Effect of The Fusion Layer
## In-depth Analysis
# Conclusion

Current Knowledge Graphs (KGs) alone are often insufficient for capturing complex, multi-hop structural relationships in critical domains
# References
1. Hager P, Jungmann F, Holland R, et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making[J]. Nature medicine, 2024, 30(9): 2613-2622. ^5050c9
2. Li Y, Li Z, Zhang K, et al. Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge[J]. Cureus, 2023, 15(6). ^81499b
3. Wu J, Zhu J, Qi Y, et al. Medical graph rag: Towards safe medical large language model via graph retrieval-augmented generation[J]. arXiv preprint arXiv: 2408.04187, 2024. ^8f364a
4. Wen Y, Wang Z, Sun J. MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024: 10370-10388. ^8a6a41
5. Edge D, Trinh H, Cheng N, 等. From Local to Global: A Graph RAG Approach to Query-Focused Summarization[A]. arXiv, 2025. ^37fab8
6. Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks[J]. Advances in neural information processing systems, 2020, 33: 9459-9474. ^953bea
7. Kim J, Kwon Y, Jo Y, et al. KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models[C]//2023 Findings of the Association for Computational Linguistics: EMNLP 2023. Association for Computational Linguistics (ACL), 2023: 9410-9421. ^0e66d4

# Appendix
