# Paper

## Main-text methods summary

We developed DnaRna, an AI model for predicting potential DNA-RNA interactions. DnaRna uses a two-tower representation and pairwise fusion architecture. It first standardizes DNA and RNA sequences and, when necessary, segments long sequences into fixed-length windows to satisfy the input constraints of pretrained encoders. It then independently encodes DNA and RNA sequences using DNABERT-2 and RNA-FM, respectively, to obtain high-dimensional representations. These modality-specific representations are subsequently fused at the pair level, and a multilayer perceptron outputs the binding probability for a given DNA-RNA pair. For large-scale candidate search, we further incorporate an optional single-sequence prescreening step to reduce the combinatorial search space while retaining discriminative performance and improving inference efficiency.

## Supplementary methods

### 1. Task definition

We formulate the task as supervised DNA-RNA pair prediction. For any given DNA sequence $d$ and RNA sequence $r$, the model outputs a scalar probability $p(y=1 \mid d,r)$, representing the likelihood that the sequence pair forms a functional binding event or interaction.

### 2. Model architecture

DnaRna uses a two-tower representation and pairwise fusion framework. The overall workflow comprises four steps: sequence preprocessing, modality-specific representation learning, optional candidate prescreening, and DNA-RNA pair classification. DNA and RNA sequences are first passed independently through their respective encoders to obtain fixed-dimensional sequence-level representations. These representations are then fused at the pair level and fed into a classification head for binary prediction. For applications that require searching large sequence libraries for candidate interactions, we allow an optional single-sequence top-K prescreening step before pairwise scoring to reduce the number of DNA-RNA combinations that must be explicitly evaluated.

#### 2.1 Modality-specific representation learning for DNA and RNA

The DNA branch uses DNABERT-2, a pretrained foundation model for DNA sequences, as the encoder. Within each DNA window, the sequence is first tokenized and tensorized, after which information is extracted from the final-layer representation and aggregated into a fixed-length sequence embedding $h_d$. This design avoids additional complex aggregation modules while retaining the pretrained model's ability to encode local nucleotide context.

The RNA branch uses RNA-FM, a pretrained foundation model for RNA sequences, as the encoder. After standardization, RNA sequences are fed into the official RNA-FM model, and information is extracted from the final-layer representation and aggregated into an RNA sequence embedding $h_r$.

Accordingly, each DNA window and RNA window is ultimately represented as a fixed-dimensional vector, $h_d \in \mathbb{R}^{D_d}$ and $h_r \in \mathbb{R}^{D_r}$, respectively. These representations are then used for both single-sequence scoring and pairwise prediction.

#### 2.2 Optional candidate prescreening

When the number of DNA or RNA candidates is large, directly scoring all $N_{\mathrm{DNA}} \times N_{\mathrm{RNA}}$ combinations incurs substantial computational cost. To address this issue, we introduce an optional single-sequence prescreening module. This module applies a binary classifier separately to DNA and RNA embeddings, assigns a prediction score to each sequence, and retains the top-K candidates ranked by score. Pairwise prediction is then carried out only within the retained candidate subsets.

This prescreening procedure serves scalability rather than changing the discriminative form of the pairwise model itself. In other words, when top-K is disabled, the model can still evaluate all DNA-RNA combinations directly; when top-K is enabled, the candidate space can be reduced from $N_{\mathrm{DNA}} \times N_{\mathrm{RNA}}$ to $K_{\mathrm{DNA}} \times K_{\mathrm{RNA}}$, thereby substantially lowering the cost of pair enumeration and inference. For example, if the candidate pool contains $10^4$ DNA sequences and $10^4$ RNA sequences, exhaustive evaluation requires scoring $10^8$ combinations; if prescreening retains the top $10^3$ DNA and top $10^3$ RNA candidates, the number of combinations to be evaluated is reduced to $10^6$, corresponding to a two-order-of-magnitude reduction.

#### 2.3 DNA-RNA pair representation and classification head

During pairwise scoring, pair features are constructed from the sequence-level embeddings of DNA and RNA. Let the DNA and RNA representations be denoted by $h_d$ and $h_r$, respectively. In this study, we obtain the pair feature $h_{\mathrm{pair}}$ by vector concatenation:

$$
h_{\mathrm{pair}} = [h_d; h_r].
$$

This concatenation preserves modality-specific information from both DNA and RNA while providing a unified input representation for downstream classification.

After obtaining $h_{\mathrm{pair}}$, we use a multilayer perceptron as the classification head. The network contains two hidden layers of dimensions 512 and 256, respectively, with ReLU activations between layers, followed by a scalar output layer that produces a logit which is mapped to a probability by a sigmoid function.

### 3. Data preparation

#### 3.1 Sequence preprocessing and windowing

Prior to model training, we first cleaned the raw DNA-RNA pair table together with the corresponding DNA and RNA sequence tables. This step removes duplicate pairs, duplicate or invalid sequences, and pair records lacking matched DNA or RNA entries, thereby improving the consistency and usability of the downstream training data. The retained DNA and RNA sequences are then standardized. DNA sequences are converted to uppercase and restricted to the alphabet A/C/G/T/N. RNA sequences are also converted to uppercase, and T is replaced with U before input to RNA-FM; the retained alphabet is A/U/C/G/N. This preprocessing reduces formatting heterogeneity and ensures that sequences from different sources can be consistently passed to the pretrained encoders.

For sequences longer than the allowable encoder input range, we use sliding-window segmentation. Given an original sequence $s$, a window length $L$, and a stride $\Delta$, the sequence is represented as an ordered set of subsequences $\{w_i\}_{i=1}^{n}$:

$$
\{w_i\}_{i=1}^{n} = \mathrm{Window}(s; L, \Delta).
$$

In the current implementation, sequences of length no greater than $L$ are kept unchanged; sequences longer than $L$ are segmented into overlapping windows with a fixed stride, and the start and end positions of each window, together with its source sequence identity, are recorded. This strategy maps local sequence contexts to fixed-length representations without truncating the entire long sequence. In the practical training workflow used in this study, both DNA and RNA sequences were segmented using a window length of 1000 nt and a stride of 500 nt. The resulting DNA and RNA windows were then expanded jointly with pair annotations to form a window-level dataset for downstream embedding extraction and pairwise training.

#### 3.2 Construction of positive and negative samples

The positive DNA-RNA pairs used in this study were derived primarily from the NPInter database. This resource contains a large number of reported DNA-RNA interaction records together with the corresponding DNA and RNA sequence information, and therefore provides a suitable basis for positive training examples. Because there is currently no widely accepted dataset of DNA-RNA pairs that can be confidently defined as non-interacting, high-confidence negatives cannot be obtained directly. We therefore constructed negatives by randomly recombining DNA and RNA identifiers, while ensuring that the generated combinations did not overlap with the known positive set. Let $N_{+}$ denote the number of positive samples and $r$ the negative sampling ratio; the target number of negatives is then $N_{-}=rN_{+}$. In the training configuration used here, we set $r=1.0$, corresponding to a 1:1 positive-to-negative ratio.

It should be noted that such randomly generated negatives are biologically weak negatives, because some of them may in fact correspond to true positives that have not yet been experimentally validated. Accordingly, performance estimates obtained under this setting should be interpreted primarily as measures of ranking and discrimination rather than as a strict characterization against a definitive biological negative set. In other words, DnaRna is better viewed as an in silico prescreening tool for prioritizing high-scoring DNA-RNA candidates from large search spaces; these predictions still require downstream experimental validation.

### 4. Training strategy

#### 4.1 Data split

In this study, samples were randomly split into training and validation sets at a ratio of 9:1, with 90% of the samples used for model training and the remaining 10% used for validation and identification of the best-performing model. Before splitting, the samples were cleaned and deduplicated to improve consistency between the training and validation data.

#### 4.2 Feature standardization and optimization

Before entering the classification head, pairwise features are standardized using statistics estimated from the training set as follows:

$$
\hat{x} = \frac{x - \mu}{\sigma},
$$

where $\mu$ and $\sigma$ denote the mean and standard deviation of the corresponding feature dimension in the training set.

The model is optimized using binary cross-entropy with logits, defined as:

$$
\mathcal{L} = - \left[ y \log \sigma(z) + (1-y)\log \left(1-\sigma(z)\right) \right],
$$

where $y \in \{0,1\}$ denotes the ground-truth label, $z$ denotes the model output logit, and $\sigma(\cdot)$ denotes the sigmoid function. Parameter updates are performed using the AdamW optimizer.

During training, logits are converted to probabilities through a sigmoid mapping, and 0.5 is used as the default classification threshold. The best-performing model can be retained according to validation performance, and early stopping can be optionally enabled.

### 5. Inference workflow

#### 5.1 Pairwise scoring

During inference, the same pairwise feature construction strategy used in training is retained. Given collections of DNA and RNA embeddings, the model can either score pre-specified candidate pairs or perform exhaustive pairwise prediction over all DNA-RNA combinations, outputting both the predicted probability and binary label for each pair. In the practical screening workflow used in this study, we performed exhaustive scoring over the full combinatorial set of input DNA and RNA embeddings. For example, if the candidate pool contains $10^4$ DNA sequences and $10^4$ RNA sequences, the model evaluates $10^8$ DNA-RNA combinations.

In large-scale inference settings, pairwise features are constructed in blocks and forward passes are executed in batches to control GPU and host-memory usage. If candidate prescreening is enabled, the model performs pairwise prediction only within the retained DNA and RNA candidate subsets, thereby reducing the number of combinations that must be evaluated (see Section 2.2); otherwise, all candidate combinations are processed directly. Final outputs can include sample identifiers, predicted probabilities, and thresholded labels for each pair.

#### 5.2 Pair-level aggregation of window-level predictions

Because long DNA or RNA sequences may be segmented into multiple windows before encoding, a single original DNA-RNA pair can correspond to multiple window-level combinations during inference. To convert window-level predictions into a ranking score at the original sequence-pair level, we aggregate all window-pair predictions derived from the same original DNA and RNA sequences.

Let the original DNA sequence $d$ be segmented into $m$ windows and the original RNA sequence $r$ into $n$ windows. Let $p_{ij}$ denote the predicted interaction probability for the window pair $(i,j)$. We compute the overall interaction score for the original DNA-RNA pair using a noisy-or aggregation:

$$
S(d,r) = 1 - \prod_{i=1}^{m}\prod_{j=1}^{n}(1-p_{ij}).
$$

This score is used to rank candidate DNA-RNA pairs at the original sequence level. Compared with simple averaging, noisy-or aggregation emphasizes the case in which at least one local window pair carries a strong interaction signal, making it suitable for candidate prioritization after long-sequence windowing. When a binary decision is required, the aggregated score can be further converted into a pair-level prediction label using a predefined threshold.

In addition to the overall interaction score, we record auxiliary interpretability measures, including the maximum window-level probability, the mean window-level probability, the number and fraction of window pairs above the threshold, and the numbers of DNA, RNA, and total window pairs. These measures are not treated as independent model performance metrics; instead, they help interpret the aggregated score, localize the window regions contributing most strongly to the prediction, and assess the influence of window count on the aggregated result.

### 6. Discussion

#### 6.1 Strengths of the method

The main strength of this framework lies in its combination of the representational power of pretrained biological sequence models with the efficiency of a lightweight discriminator. This design allows the model to exploit rich sequence priors while remaining stable to train with limited data. The two-tower design allows DNA and RNA to be modeled by separate pretrained backbones that are better suited to each modality, whereas the pair classification head restricts cross-modal fusion to a small and controllable parameter space.

#### 6.2 Limitations of the method

This approach also has several limitations. First, because negatives are generated by random recombination, the training set may contain unannotated positives, which can affect probability calibration and absolute performance estimates. Second, the windowing strategy improves tractability for long sequences but weakens the modeling of long-range dependencies across windows. Third, current pair fusion relies primarily on first-order interactions at the vector level and does not explicitly model higher-order cross-modal attention or structural constraints. Accordingly, this framework is better viewed as a robust and scalable baseline or practical screening system than as a definitive description of the molecular mechanism of interaction.

#### 6.3 Significance of the model

The primary significance of this model is that it provides a practical computational framework for large-scale discovery of potential DNA-RNA interactions. If a candidate pool contains $D$ DNA sequences and $R$ RNA sequences, the potential search space can reach $D \times R$. At this scale, experimentally testing every combination one by one is typically infeasible in terms of cost, labor, and time. In the absence of an effective prior model, candidate pairs would often have to be sampled in an approximately random manner, resulting in a low hit rate for true interactions. By contrast, DnaRna can rank large candidate sets and prioritize DNA-RNA pairs with a higher predicted probability of interaction, thereby concentrating experimental effort on a smaller and more informative candidate space. In this sense, the model can be regarded as an AI-enabled screening strategy.

At the same time, the outputs of DnaRna should be interpreted as computational predictions of potential interactions rather than definitive evidence of true biological interactions. Because the current training data are derived mainly from public-database positives and randomly constructed weak negatives, the model score cannot be taken as a direct substitute for biological evidence and still requires rigorous downstream experimental validation. Even so, in this study, candidates prioritized by the model led to the experimental identification of new DNA-RNA pairs and their associated mechanisms, yielding new biological insights and providing clues for potential therapeutic strategies. This indicates that the model has clear practical value in real research workflows.

In addition, current model training relies primarily on public data resources, which have limitations in completeness, accuracy, and noise control. In future work, we will continue to accumulate high-quality, rigorously validated DNA-RNA interaction data through more direct experimental approaches, with the aim of further improving model generalization, ranking performance, and the reliability of its outputs. As higher-quality datasets continue to expand, DnaRna can be iteratively refined and is expected to play an increasingly important role in the discovery and mechanistic study of DNA-RNA interactions.
