## Frequently asked ML questions

### Q: What is the assumption of linear regression?
A: True data distribution is $p(y_i|x_i, w)=\mathcal{N}(w^Tx_i, \sigma)$ where $\sigma$ is the same for all $i$. 
Almost no linear dependency between features (because if there is, $w\approx(X^TX)^{-1}X^Ty$ explodes).
Also, all $x_i$ are independent.

### Q: What is the Bias-Variance tradeoff?
A: "Bias" here is $E(prediction-true)$, "Variance" here is the variance of predictions. High bias is a symptom of underfitting, high variance may be a symptom of overfitting. It is possible to decompose the MSE error to $Bias^2+Variance+\sigma^2$, where $\sigma^2$ is the irreducible error.

### Q: How do I know the training dataset is bad?
A: Large gap remains between train and validation losses.

### Q: How do I know the validation dataset is bad?
A: Validation loss is noisy. Also, if the whole validation loss curve lies below the training loss curve, this means the validation dataset is too easy to predict.

### Q: Why we standardize the data and when?
A: Naive answer is, we standardize the features that are measured in different units to treat them equally. Tree-based methods don't need standardization. 
Note, that StandardScaler and MinMaxScaler are highly sensitive to outliers (there is RobustScaler, which is less sensitive)

### Q: How decision trees work?
A: On each node we choose the splitting criteria with max information gain, that is $H_{parent}-\sum_i w_i H_{child}$, 
where $H_{parent}$ is an entropy of the class distribution in the parent (current) node, and $H_{child}$ – in the $i$' th child node; $w_i=\frac{N_{child}}{N_{parent}}$, 
the fraction of object counts. For regression, predicted value is the mean value of the corresponding leaf distribution, and instead of entropy $H$ the variance is used.

### Q: What is model Blending?
A: Ensemble method. Train base models on train set, predict on validation set, then train a meta-model on predictions from validation set. Main disadvantage: base models don't see the full training set. Improvement: we can perform multiple base models training (on different splits of the train set), concatenate their predictions and train the meta-model on this bigger set.

### Q: What is model Stacking?
A: Ensemble method. "Improved Blending + KFold". Train base models on all folds except one, predict on this fold -> we got meta-features on the whole train set. Main disadvantage: small folds may result in very different learning problems. Improvement: perform multiple kfolds, concatenate meta-features. People advise using algorithms of different nature for obtaining meta-features.

### Q: What is model Bagging?
A: Ensemble method. Bootstrap aggregation. Sample with replacement from training set, train a base model on this bootstrap sample. Average the predictions of base models for a final prediction. We can measure the efficiency of bagging via out-of-bag error, that is an average error of samples, not presenting in a training sample for each model.

### Q: What is Random Forest?
A: Ensemble method: CART + Bagging + (maybe) random subsets of features. Build trees from bootstrap subsamples and feature subsamples. One can also set max tree depth and min amount of samples in leafs.

### Q: What is model Boosting?
A: Ensemble method, we train a sum of models ("weak learners") recursively, such that next model is trained to minimize the partial sum. 

### Q: What is Gradient Boosting?
A: Ensemble method. Each weak learner $a_i$ is trained on $(x_i, -\nabla L(y_i, a_{i-1}(x_i)))$. Modifications: add learning rates $\lambda_i$, add bagging (learn on subsamples on each step).

### Q: What is a Kernel Trick?
A: A simple feature extraction technique is applying a non-linear transformation to objects. However, for big number of features this may be hard in general, as we may need to add all possible feature combinations. Instead, we might introduce a function that compares two objects in the "old" feature space: a kernel $k(x,x')$. The "typical" kernel model is $\hat y(x) = \sum_i L(w_i, y_i, k(x_i,x))$.

### Q: Name binary classification metrics
A: Let P – positive, N – negative, T – true classified, F – false classified.
Then:
| metric | formula |
|--------|---------|
| precision | $\frac{TP}{TP+FP}$ |
| sensitivity (recall, true positive rate) | $\frac{TP}{P}$  |
| false positive rate | $\frac{FP}{N}$ |
| specificity (true negative rate) | $\frac{TN}{N}$ |
| accuracy | $\frac{TP+TN}{P+N}$ |
| f1 | $2\frac{precision\cdot recall}{precision+recall}$ |
| geometric mean | $\sqrt{recall\cdot specificity}$ |

High sensitivity/recall means we catch the positive class well, but maybe we misclassify negative objects. High precision means the objects we mark as positive are likely positive, but maybe we mark very little positive objects. High false positive rate means we have a lot of misclassified negative objects, high specificity means we have a lot of correctly classified negative objects. For multiclass classification or even regression we might want to plot the confusion matrix where $a_{ij}$ is the number of objects that belong to the class $i$ and classified to the class $j$.

For binary classification that predicts class probability we need to invent a threshold for making the final decision. In this case it is convenient to draw a curve $c(x,y,threshold)$:

| curve | y | x | optimal threshold |
|--------|---------|----------|-------|
| ROC | true positive rate (recall, sensitivity) | false positive rate (1-specificity) | max gmean |
| DET | false negative rate | false positive rate | max gmean |
| precision-recall curve | precision | recall | max f1 |

### Q: For multiclass classification, what is micro/macro/weighted averaging of a metric?

A: For instance for $precision=\frac{TP}{TP+FP}$ micro average is the $\frac{TP_{all\\;classes}}{TP_{all\\;classes}+FP_{all\\;classes}}$, and macro average is $mean(precision_{class_1},precision_{class_2},...)$. Weighted average is macro average where each classwise metric is weighted by $class\\;samples\\;/\\;all\\;samples$.

for **micro** all **samples** equally contribute to the metric

for **macro** all **classes** equally contribute to the metric

for **weighted** the contribution of each class is proportional to its size

### Q: Name learning to rank metrics
A: Learning to rank is predicting the relevance score given a pair (query, document).

**Average Precision.** Shrink top-k documents for a particular query and compute precision/recall (target = 0, "irrelevant", or 1, "relevant"). For different k we get a precision recall curve. The area under the curve is the Average Precision score.

**Mean Average Precision.** Compute average precisions for all queries, take a mean.

**Cumulative Gain / Graded Precision.** Applicable if target is a graded relevance. For a specific query, if answers are sorted by graded relevance, CG is a sum of first $p$ relevances: $CG={\sum}^p_{i=1} rel_i$

**Discontinued Cumulative Gain.** CG with logarithmic reduction factor. Two formulations: $DCG_1={\sum}^p_{i=1}\frac{rel_i}{\log_2(i+1)}$ and $DCG_2={\sum}^p_{i=1}\frac{2^{rel_i}-1}{\log_2(i+1)}$ 

**Normalized Discontinued Cumulative Gain.** DCG divided by max possible DCG for this query

**Mean CG / DCG / NDCG.** Average for all queries.

**Mean reciprocal rank.** Applicable to 1/0 target. For all queries $q_i\in Q$: $MRR=\frac{1}{|Q|}{\sum_i}^{|Q|} \frac{1}{rank_i}$, where $rank_i$ is the position of the first relevant document for query $q_i$

### Q: How to fine-tune models?
A: Training the new model head (randomly initialized) is usually performed with big learning rate (other weights need to be frozen). The good weights are usually fine-tuned with small learning rates.

### Q: Data Augmentation techniques

CV:
1. **Crop.**
2. **Affine transformation.** Rotation, zoom, mirror, etc.
3. **Noise injection.**
4. **Masking.**
5. **Image properties change.**

NLP:
1. **Inject misspellings / typos / OCR errors**
2. **Replace some words with synonyms / predicted from context.**
3. **Delete random words (depending on the POS, for instance).**

Timeseries / Audio:
1. **Noise injection.**
2. **Time shift / speed change.**
3. **Change of spectrum.** Pitch, EQ, Echo, masking, etc: for Fourier-based data

### Q: Name clustering metrics 

**Silhouette score.** Let $b(i)$ be the mean distance between point $i$ and other points in its cluster, and $a(i)$ the min average distance between point $i$ and points in other clusters. The Silhouette score for point $i$ is $\frac{b(i)-a(i)}{\max (b(i), a(i))}$. The overall score is averaged over all points. It is common to do a **silhouette analysis,** that is, plot silhouette coeffitients for each sample along x axis and cluster sizes along y axis.

**Homogeneity and Completeness.** Define $p(c,k)$ a probability of observing label $c$ in cluster $k$, $p(k)$ the probability of a point being in the cluster $k$ and $p(c)$ the probability of a point having a label $c$. If $H(c|k)$ is their relative entropy, then the Homogeneity score is $homogeneity=1-\frac{H(c|k)}{H(c)}$. The completeness score is the opposite: $completeness=1-\frac{H(k|c)}{H(k)}$. Max homogeneity means if two points are in the same cluster, they have the same label. Max completeness means if two points have the same label, they are in the same cluster.

**V-Measure or Normalized Mutual Information.** It is an F1-like normalized homogeneity and completeness: $V=2\frac{homogeneity\cdot completeness}{homogeneity+completeness}$

---

# Unresolved

### Q: Explain classical Bayesian inference
A: If we have a $\theta$-parameterized statistical model of our answers $p(y|x,\theta)$, we can use the training sample $(x,y)$ to fit the $\theta$ via the Bayes theorem: $p(\theta|x,y)=\frac{p(y|x,\theta)p(\theta)}{p(y)}$. Next we can do a Maximum a posteriori estimation $\theta = \arg\max_\theta p(y|x,\theta)p(\theta)$, but the full inference is deriving the density of $p(\theta|x,y)$, i.e. computing the integral $\int p(y|x,\theta)p(\theta) d\theta$ which is generally impossible to do analytically. However, if $p(\theta)$ and $p(y|theta)$ are *conjugate* distributions, then the posterior $p(\theta|y,x)$ belongs to the same family as $p(\theta)$.

### Q: What is distribution entropy, conditional entropy and perplexity?
A: Shannon Entropy is defined as $H(x) = E_{x\sim p}(-\log p(x)) = - \int p(x)\log p(x) dx $

Conditional Entropy: $H(x|y) = E_{x,y\sim p_{x,y}} -\log\frac{p_{x,y}(x,y)}{p_{y}(y)} = - \int p_{x,y}(x,y)\log\frac{p_{x,y}(x,y)}{p_{y}(y)}$

Perplexity measure is defined as $2^H(p)$

### Q: What is cross-entropy?
A: Cross-entropy is defined as $H(p,q)=E_{x\sim p}-\log q(x)=-\int p(x)\log q(x)$. It is used as a loss function with $p$ is a true distribution, and $q$ is a predicted distribution.

### Q: What are KL and JS-divergences?
KL-divergence or relative entropy is $KL(p||q)=E_{x\sim p}(\log\frac{p(x)}{q(x)}) = \int p(x)\log\frac{p(x)}{q(x)} dx = H(p,q)-H(p)$. JS-divergence is a symmetrized KL: $JS(p,q) = \frac{1}{2}(KL(p||\frac{p+q}{2})+KL(q||\frac{p+q}{2}))$

### Q: What are the exploding gradients and vanishing gradients problems?
A: Exploding gradients means $|\frac{\partial L}{\partial w}|\rightarrow\infty$. Can be avoided with gradient clipping, choosing a smaller learning rate, layer output normalization. Vanishing gradients is when $|\frac{\partial L}{\partial w}|\rightarrow 0$. It is often caused by constant parts of the activation functions. Also, to overcome exploding / vanishing gradients, a proper initialization should be used.

### Q: Explain dropout
A: A regularization layer. During training zeroes the outputs with probability $p$ and scales them by $\frac{1}{1-p}$. Dropout is usually placed after the activation and it should have lower $p$ values after convolutional layers. After SELU activations the *alpha dropout* is used as it doesn't shift the mean/variance after them.

### Q: What weight initialization to choose?
A: They ensure the variance of the activated outputs is the same as the variance of the inputs for linear layers. Xavier initialization $N(0, \frac{1}{n_{inputs}})$ works for linear, sigmoid and tanh. He initialization $N(0, \frac{2}{n_{inputs}})$ works for ReLU. $N(0,\sqrt{\frac{1}{n_{inputs}}})$ works for SELU.

### Q: Explain normalization techniques

**Batch Normalization** is applying the $\gamma \frac{x-\mu_x}{\sigma_x} + \beta$ transformation along the batch axis where $\gamma$ and $\sigma$ are trainable parameters. It standardizes the output in batch, makes the network train faster and acts as a regularizer.

**Layer Normalization** is the same but along layer outputs. Prevents exploding / vanishing gradients. It is used in the transformer architecture.

**Instance normalization** is the same along each sample.

**Adaptive Instance Normalization** is used for style transfer: $adain(x,y)=\sigma(y)\frac{x-\mu_x}{\sigma_x}+\mu_y$.

### Q: Explain attention techniques
A: First, attention is a function of the query and the set of key-value pairs $attention(Q,K,V)=align(Q,K)V$. For scaled dot product attention $score(Q,K)=softmax(QK^T / \sqrt{d_K})$. In the encoder-decoder attention $K$, $V$ are from encoder, and $Q$ is from decoder. In self-attention they are all the same.

### Q: What is a softmax?
A: It is a normalized vector exponent $softmax(x)_i=\frac{e^x_i}{\sum_i e^x_i}$, it is used to transform the outputs to a normalized histogram.

### Q: Explain early stopping
A: Stop gradient descent if a selected metric stops getting better. A validation metric must be used.

### Q: Give a quick recap of most useful statistical tests

**Chi-square test**

Used for determining the difference between distributions: $\chi^2 = \sum_{sample}\frac{(observed-true)}{true}$. Small $p$ value means stronger difference.

**t-test**

Used to compare the means of two distributions. Lower $p$ values correspond to stronger difference.

**Shapiro-Wilk test**

Used to check if the data is sampled from a normal distribution. Lower $p$ values mean it is not.

### Q: Markov and Chebyshev inequalities

$P(x\geq a)\leq \frac{Ex}{a}$

$P(|x-Ex|\geq k\sigma)\leq \frac{1}{k^2}$





