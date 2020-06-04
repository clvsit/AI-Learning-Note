为了加快推理速度并减小模型大小，并同时保持精度，作者提出了一种新颖的 Transformer 的模型知识蒸馏（KD）设计。通过利用这种新的 KD 方法，可以将大型教师模型（例如 BERT）中编码的大量知识很好地转移到小型学生模型 TinyBERT 中。

此外，作者为 TinyBERT 引入了一个新的两阶段学习框架，该框架在预训练阶段和特定于任务的学习阶段均执行 Transformer 蒸馏。该框架确保 TinyBERT 可以捕获 BERT 中的通用领域以及特定于任务的知识。

TinyBERT 在 GLUE 基准测试中，其性能达到了教师模型 BERT-Base 的 96% 以上，但内存占用小 7.5 倍，推断速度快 9.4 倍。TinyBERT 还比 BERT 蒸馏的最新 baseline 好得多，仅需要 28% 的参数，但推理速度提升 31%。

## 1. 介绍
预训练语言模型，然后对下游任务进行微调已成为自然语言处理的新范例。预训练语言模型（PLM），例如 BERT、XLNet、RoBERTa 和 SpanBERT 在许多 NLP 任务和具有挑战性的 multi-hop 推理任务中都取得了巨大的成功。但是，PLM 通常具有大量的参数，并且需要较长的推断时间，因此很难在移动设备上进行部署。此外，最近的研究也证明了 PLM 中存在冗余。因此，在保持性能的同时减少 PLM 的计算开销和模型存储至关重要且可行。

目前已有许多模型压缩技术，以加速深层模型推断并减小模型大小，同时保持精度。最常用的技术包括量化、权重剪枝和知识蒸馏（KD）。在本篇论文中，作者专注于知识蒸馏，这是 Hinton 等人在教师模型和学生模型框架中提出的一个想法。KD 旨在将 embedding 在大型教师模型中的知识转移到小型学生模型中，训练学生模型以重现教师模型的行为。

在此框架的基础上，作者针对基于 Transformer 的模型提出了一种新颖的蒸馏方法，并以 BERT 为例研究了大型 PLM 的 KD 方法。在 NLP 中对 KD 进行了广泛的研究，但为 BERT 设计 KD 方法的研究较少。首先在大型无监督文本语料库上对 BERT 进行预训练，然后在任务特定的数据集上对其进行微调，这极大地增加了 BERT 蒸馏的难度。因此，我们需要为这两个阶段设计有效的 KD 策略。为了构建具有竞争力的 TinyBERT，作者首先提出了一种新的 Transformer 蒸馏方法，以提取 BERT 教师模型中嵌入的知识。

具体来说，作者设计了几种损失函数以适应 BERT 层的不同表征：
- 嵌入层的输出；
- 来自 Transformer 层的隐状态和 attention 矩阵；
- 预测层输出的 logits。

基于注意力的拟合得益于最近的研究发现，即 BERT 所学习的注意力权重可以捕获大量的语言知识，从而将语言知识可以很好地从 BERT 教师模型蒸馏给 TinyBERT 学生模型。但是，在 BERT 的现有 KD 方法中被忽略了（BERT-PKD，DistillBERT 等）。

除了设计损失函数之外，作者提出了一种新颖的两阶段学习框架，包括通用蒸馏和特定任务蒸馏。
- 在通用蒸馏阶段，未经微调的原始 BERT 充当教师模型。TinyBERT 学生模型通过在通用领域对大型语料库执行提出的 Transformer 蒸馏方法来学习模仿教师模型的行为，从而获得了可以对各种下游任务进行微调的通用 TinyBERT。
- 在特定任务的蒸馏阶段，作者执行数据增强以提供更多特定任务的数据供教师和学生模型学习，然后在增强的数据集上重新执行 Transformer 蒸馏。

这两个阶段对于提高 TinyBERT 的性能和泛化能力都是必不可少的。表 1 中总结了提出的方法与其他现有方法之间的详细比较。Transformer 蒸馏和两阶段学习框架是提出方法的两个关键思想。

![Table 1](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Table%201.jpg)

本篇论文的主要贡献如下：
1. 提出了一种新的 Transformer 蒸馏方法，将 BERT 教室模型编码的语言知识转移到 TinyBERT。
2. 提出了一个新颖的两阶段学习框架，该框架可以在预训练和微调阶段执行提出的 Transformer 蒸馏，从而确保 TinyBERT 可以捕获教师模型的一般领域知识和特定于任务的知识。
3. 通过实验证明，TinyBERT 可以在 GLUE 任务上实现超过 BERT-Base 教师模型 96% 的性能，同事具有更少的参数和更短的推断时间，并且明显优于其他现有 BERT 蒸馏技术基准。

## 2. 前提条件
首先介绍 Transformer 和 Knowledge Distillation 公式，因为作者提出的 Transformer 蒸馏是针对 Transformer 模型特别设计的一种 KD 方法。

## 3. 方法
在本节中，作者重点介绍了提出的蒸馏方案。

### 3.1 Transformer 蒸馏
图 1 显示了蒸馏方法的全貌，在这项工作中，学生和教师模型都使用 Transformer 构建。

![Figure 1](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Figure%201.jpg)

假设学生模型具有 M 个 Transformer 层，教师模型具有 N 个 Transformer 层，作者从教师模型中选择 M 层进行 Transformer 蒸馏。函数 n = g(m) 用作从学生模型层到教师模型层的映射函数，这意味着学生模型的第 m 层从教师模型的第 n 层学习知识。

除了 Transformer 蒸馏外，还考虑了嵌入层蒸馏和预测层蒸馏。作者将 0 设为嵌入层的索引，将 M + 1 设为预测层的索引，并将相应的层映射分别定义为 0 = g(0) 和 N + 1 = g(M + 1)。实验部分将研究不同映射函数对性能的影响。一般地，学生模型可以通过最小化以下目标从教师模型那里获得知识。
```math
\mathcal{L}_{\text {model }}=\sum_{m=0}^{M+1} \lambda_{m} \mathcal{L}_{\text {layer }}\left(S_{m}, T_{g(m)}\right) \quad (6)
```

其中 `$\mathcal{L}_{layer}$` 是指给定模型层（例如，Transformer 或嵌入层）的损失函数，而 `$\lambda_m$` 表示第 m 层蒸馏重要性的超参数。

#### Transformer 层蒸馏
Transformer 层蒸馏包括基于注意力的蒸馏和基于隐状态的蒸馏，如图 1(b) 所示。基于注意力的蒸馏是受最近研究的启发，即可以从 BERT 学习到的注意力权重中捕获丰富的语言知识。这种语言知识包括语法和共指信息，这对于自然语言理解至关重要。具体来说，学生模型学习教师模型中的多头注意力矩阵，目标函数定义为：
```math
\mathcal{L}_{\mathrm{attn}}=\frac{1}{h} \sum_{i=1}^{h} \operatorname{MSE}\left(\boldsymbol{A}_{i}^{S}, \boldsymbol{A}_{i}^{T}\right) \quad (7)
```

其中，h 是注意力头的数量，`$A_i \in \R^{l \times l}$` 表示对应于第 i 个教师模型或学生模型的头的注意力矩阵，l 是输入文本的长度，MSE() 表示均方误差损失函数。在本篇论文中，未归一化的注意力矩阵 Ai 被当作拟合目标，而不是 softmax 层输出的 softmax(Ai)，因为实验表明，前一种设置具有更快的收敛速度和更好的性能。

除了基于注意力的蒸馏外，作者还从 Transformer 层的输出中提取知识，目标函数如下：
```math
\mathcal{L}_{\mathrm{hidn}}=\operatorname{MSE}\left(\boldsymbol{H}^{S} \boldsymbol{W}_{h}, \boldsymbol{H}^{T}\right) \quad (8)
```

其中，矩阵 `$H^S \in \R^{l \times d'}$`、`$H^T \in \R^{l \times d}$` 分别表示学生模型和教师模型的隐状态，通过等式 4 计算。标量值 d' 和 d 表示学生模型和教师模型的隐状态大小，通常 d' 小于 d 以获得更小的学生模型。矩阵 `$W_h \in \R^{d' \times d}$` 是可学习的线性变换，它将学生模型的隐状态转换为与教师模型隐状态相同的空间。

#### 嵌入层蒸馏
嵌入层蒸馏类似于基于隐状态的蒸馏，其公式为：
```math
\mathcal{L}_{embd} = MSE(E^SW_e, E^T) \quad (9)
```

其中矩阵 `$E^S$` 和 `$E^T$` 分别表示学生模型和教师模型的嵌入，在本篇论文中，它们具有与隐状态矩阵相同的 shape。矩阵 `$W_e$` 是线性变换，起着与 `$W_h$` 相似的作用。

#### 预测层蒸馏
除了模仿中间层的行为，作者还使用 KD 来拟合教师模型的预测（Hinton 等人的方法）。具体来说，对学生模型的 logit 和教师模型的 logit 之间的软交叉熵损失进行了惩罚。
```math
\mathcal{L}_{pred} = -softmax(Z^T) \cdot log\_softmax(z^S/t) \quad (10)
```

其中 `$Z^S$` 和 `$Z^T$` 分别是学生模型和教师模型预测的对数向量，log\_softmax() 表示对数似然，t 表示温度值。在作者的实验中，作者发现 t = 1 时表现良好。

使用上述蒸馏目标（即方程式 7、8、9 和 10），作者统一教师模型和学生模型之间相应层的蒸馏损失：
```math
\mathcal{L}_{\text {layer }}\left(S_{m}, T_{g(m)}\right)=\left\{\begin{array}{ll}
\mathcal{L}_{\text {embd }}\left(S_{0}, T_{0}\right), & m=0 \\
\mathcal{L}_{\text {hidn }}\left(S_{m}, T_{g(m)}\right)+\mathcal{L}_{\text {attn }}\left(S_{m}, T_{g(m)}\right), & M \geq m>0 \\
\mathcal{L}_{\text {pred }}\left(S_{M+1}, T_{N+1}\right), & m=M+1
\end{array}\right.
```

在作者的实验中，首先执行中间层蒸馏（M >= m >= 0），然后执行预测层蒸馏（m = M + 1）。

### 3.2 TinyBERT 学习
BERT 的应用通常包括两个学习阶段：预训练和微调。BERT 在预训练阶段学到的大量知识非常重要，也需要被转移。因此，作者提出了两阶段学习框架，包括通用蒸馏和特定任务蒸馏，如图 2 所示。

![Figure 2](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Figure%202.jpg)

- 通用蒸馏有助于 TinyBERT 学习 BERT 教师模型中丰富的知识，这对于提高 TinyBERT 的泛化能力起着重要作用。
- 特定任务的蒸馏向学生模型传授特定于任务的知识。

通过这两步蒸馏，可以进一步缩小教师模型和学生模型之间的差距。

#### 通用蒸馏
在通用蒸馏中，作者使用原始的 BERT（无需微调）作为教师模型，并使用大型文本语料库作为学习数据。通过对来自常规域的文本执行 Transformer 蒸馏，可以获得针对下游任务进行微调的基础 TinyBERT。但是，由于隐藏层/嵌入层的大小和层数的显著减少，因此基础 TinyBERT 的性能要比 BERT 差。

#### 特定任务蒸馏
先前的研究表明，复杂的模型，经过微调的 BERT 会因针对特定领域任务的过度参数化而受到影响（《Revealing the dark secrets of bert》）。因此，小模型有可能获得与 BERT 相当的性能。为此，作者建议通过特定任务的蒸馏诞生出具有竞争力的微调 TinyBERT 模型。

在特定任务的蒸馏中，作者在扩展的特定任务的数据集上重新执行 Transformer 蒸馏（如图 2 所示）。具体来说，将经过微调的 BERT 用作教师模型，并提出一种数据修正方法来扩展特定任务的训练集。学习更多与任务相关的数据，可以进一步提高学生模型的泛化能力。在这项工作中，作者结合了经过预训练的语言模型 BERT 和 GloVe 词嵌入，以进行单词级替换来增强数据。具体来说，使用语言模型来预测单片单词的单词替换，并使用词嵌入来检索最相似的单词作为多片单词的单词替换。作者定义了一些超参数来控制句子的替换率和扩充数据集的数量，附录 A 中讨论了数据增强过程的更多详细信息。

以上两个学习阶段是相辅相成的：通用蒸馏为特定任务的蒸馏提供了良好的初始化，而特定任务的蒸馏通过专注于特定任务的知识进一步证明了 TinyBERT。尽管 BERT 和 TinyBERT 在模型大小上有很大的差距，但通过实现两阶段蒸馏，TinyBERT 可以在各种 NLP 任务中实现有竞争力的性能。

## 4. 实验
在本节中，作者评估 TinyBERT 在具有不同模型设置的各种任务上的有效性和效率。

### 4.1 模型设置
作者实例化了一个小型的学生模型（层数 M = 4，隐藏层大小 d' = 312，前馈/过滤器大小 di' = 1200，头数 h = 12），该模型共有 1450 万个参数。如果没有特别指定，则此学生模型称为 TinyBERT。原始 BERT-Base（层数 N = 12，隐藏层大小 d = 768，前馈/过滤器大小 di = 3072，头数 h = 12）作为教师模型，包含 109M 个参数。

作者使用 g(m) = 3 x m 作为层映射函数，因此 TinyBERT 从 BERT-Base 的每 3 层中学习。每层的学习权重 `$\lambda$` 设置为 1，对于 TinyBERT 的学习效果很好。

### 4.2 在 GLUE 上的实验结果
作者根据通用语言理解评估（GLUE）基准评估 TinyBERT，该基准是各种自然语言理解任务的集合。实验设置的详细信息请参见附录 B。评估结果列在表 2 中，模型大小和推断时间列在表 3 中。

![Table 2](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Table%202.jpg)

![Table 3](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Table%203.jpg)

实验结果表明：
1. 由于模型尺寸的大幅减小，BERT-Small 与 BERT-Base 之间存在较大的性能差距。
2. TinyBERT 在所有 GLUE 任务中始终优于 BERT-Small，并且平均提高 6.3%。这表明提出的 KD 学习框架可以有效地改善小型模型的性能，而与下游任务无关。
3. 即使只有约 28% 的参数和约 31% 的推断时间，TinyBERT 的性能也至少要比最新的 KD 基线（即 BERT-PKD 和 DistillBERT）高出 3.9%，见表 3。
4. 与教师模型 BERT-Base 相比，TinyBERT 的模型大小小了 7.5 倍，速度快 9.4 倍，同时保持了出色的性能。
5. TinyBERTha 具有与蒸馏 BiLSTM-Soft 相当的模型大小（大小稍大，但推断速度更快），并且在 BiLSTM 基线报告的所有任务中均获得了更好的性能。
6. 对于具有挑战性的 CoLA 数据集（预测语言可接受性判断的任务），所有蒸馏的小模型与教师模型都有相当较大的性能差距。

TinyBERT 在基线的基础上实现了重大改进，并且可以通过使用更深、更广泛的模型来捕获更复杂的语言知识来进一步提高其性能。

此外，BERT-PKD 和 DistillBERT 使用经过良好预训练的教师模型 BERT 来初始化学生模型（表 1），这使得学生模型必须保持与教师模型相同大小的 Transformer 层和嵌入层。而在 TinyBERT 的两阶段蒸馏框架中，TinyBERT 是通过常规蒸馏初始化的，因此模型的大小选择更加灵活。

### 4.3 模型大小的影响
作者评估了在几个典型的 GLUE 任务上增加 TinyBERT 模型大小可以实现多少改进，其中在 Devlin 等人的消融研究中使用了 MNLI 和 MRPC。而 CoLA 是 GLUE 中最困难的任务。具体来说，作者提出了三个结构更深的变体，它们对验证集的评估结果显示在表 4 中。

![Table 4](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Table%204.jpg)

通过表格，我们可以观察到：
1. 所有三个 TinyBERT 变体的性能都比原始的最小 TinyBERT 要好，这表明提出的 KD 方法适用于各种模型大小的学生模型。
2. 对于 CoLA 任务，仅增加层数（从 49.7 到 50.6）或隐藏层大小（从 49.7 到 50.5）时，改进很小。为了取得更大的进步，学生模型应该变得越来越深（从 49.7 到 54.0）。
3. 另一个有趣的观察结果是最小的 4 层 TinyBERT 甚至可以胜过 6 层的 BERT-PKD，这进一步证实了所提出的 KD 方法的有效性。

### 4.4 消融研究
在本节中，作者进行了消融研究，以研究一下方面的贡献：
1. 提出的两阶段 TinyBERT 学习框架的过程（图 2）；
2. 不同的蒸馏目标（公式 11）。

#### 不同学习过程的影响
提出的两阶段 TinyBERT 学习框架包含三个关键过程：TD（特定任务蒸馏），GD（通用蒸馏）和 DA（数据增强）。在表 5 中列出不同学习过程的影响。结果表明，所有三个步骤对于提出的 KD 方法都至关重要。TD 和 DA 在所有四个任务中具有可比的效果。作者还发现，在所有四个任务中，特定任务的过程（TD 和 DA）比预训练过程（GD）更有效。

另一个有趣的发现是，GD 对 CoLA 的影响大于对 MNLI 和 MRPC 的影响。作者推测 GD 学习到的语言概括能力在下游 CoLA 任务中起到更重要的作用。

#### 不同蒸馏目标的影响
作者研究了蒸馏目标对 TinyBERT 学习的影响，提出了一些基线，包括不使用 Transformer 层蒸馏的 TinyBERT（No Trm），不使用 Embedding 层蒸馏的 TinyBERT（No Emb）以及不使用预测层蒸馏的 TinyBERT（No Pred）。结果列在表 6 中，表明所有提出的蒸馏目标对于 TinyBERT 学习都是有效的。

![Table 6](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Table%206.jpg)
- 在 No Trm 设置下，性能从 75.3 下降到 56.3，这表明 Transformer 层蒸馏是 TinyBERT 学习的关键。
- 此外，作者研究了在 Transformer 层蒸馏中注意力（No Attn）和隐状态（No Hidn）的贡献。可以发现，基于注意力的蒸馏比基于隐状态的蒸馏具有更大的影响。同时，这两种知识的提升是互补的，这使得 TinyBERT 获得了竞争优势。

### 4.5 映射函数的影响
作者研究了 TinyBERT 学习中不同映射函数 n = g(m) 的影响，如第 4.1 节所述，最初的 TinyBERT 使用统一策略，并与两个典型的基线进行了比较，包括顶部策略 g(m) = m + N - M, 0 < m <= M 和底部策略 g(m) = m, 0 < m <= M，比较结果显示在表 7 中。

![Table 7](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/TinyBERT%20Table%207.jpg)

作者发现，在 MNLI 中，顶部策略的性能优于底部策略，而在 MRPC 和 CoLA 任务中则较差，这证实了以下观察结果：不同的任务以来不同 BERT 层的知识。

由于统一策略从 BERT-Base 的低层到顶层获取知识，因此在所有四个任务中，它都比其他两个基准获得了更好的性能。

## 5. 总结
在这篇论文中，作者首先介绍了一种新的 KD 方法用于基于 Transformer 的蒸馏，然后进一步提出了一个用于 TinyBERT 学习的两阶段框架。大量的实验表明，TinyBERT 在获得竞争性能的同时，显著减小了模型尺寸并缩短了原始 BERT-Base 的推断时间，这为在移动设备上部署基于 BERT 的 NLP 应用程序提供了有效的方法。