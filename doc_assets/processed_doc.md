--- PAGE 1 ---

## Attention Is All You Need

![Table](table_images/sample-table-1.png)

The table is an author and affiliation block listing contributors, their organizations, and email addresses. It presents seven authors associated primarily with Google (Google Brain and Google Research) plus one author affiliated with the University of Toronto, and it includes symbols next to some names that function as footnote markers (an asterisk “*”, a dagger “†”, and a double dagger “‡”) without showing the corresponding footnote text in the provided image.

Each entry follows the same structure: the author’s full name on one line (with an optional footnote symbol immediately after the name), the institutional affiliation on the next line (either “Google Brain,” “Google Research,” or “University of Toronto”), and an email address on the following line. The authors and details shown are: Ashish Vaswani* (Google Brain; avaswani@googlele.com), Noam Shazeer* (Google Brain; noam@google.com), Niki Parmar* (Google Research; nikip@google.com), Jakob Uszkoreit* (Google Research; uszkgoreit@google.com), Llion Jones* (Google Research; llion@google.com), Aidan N. Gomez*† (University of Toronto; aidan@cs.toronto.edu), Lukasz Kaiser* (Google Brain; lukaszkaiser@google.com), and Illia Polosukhin*‡ (email illia.polosukhin@gmail.com; no affiliation line is shown for this last entry in the image). The pattern across rows indicates most authors use Google corporate email domains (“@google.com”), one uses an academic domain (“@cs.toronto.edu”), and one uses a personal Gmail address (“@gmail.com”); one email appears to contain a likely typographical anomaly (“@googlele.com” rather than “@google.com”).



## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

## 1 Introduction

Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].

∗ Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

† Work performed while at Google Brain.

‡ Work performed while at Google Research.



--- PAGE 2 ---

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states h t , as a function of the previous hidden state h t -1 and the input for position t . This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and conditional computation [26], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [11]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8].

## 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 29]. Here, the encoder maps an input sequence of symbol representations ( x 1 , ..., x n ) to a sequence of continuous representations z = ( z 1 , ..., z n ) . Given z , the decoder then generates an output sequence ( y 1 , ..., y m ) of symbols one element at a time. At each step the model is auto-regressive [9], consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

## 3.1 Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-



--- PAGE 3 ---

Figure 1: The Transformer - model architecture.

![Figure](image_png_images/sample-picture-1.png)

The figure is a block diagram of the Transformer “encoder–decoder” neural network architecture used for sequence-to-sequence tasks. On the left is the encoder stack, shown as a repeated module labeled “N×” (meaning the same layer block is repeated N times). At the bottom of the encoder, “Inputs” are converted to an “Input Embedding,” then a “Positional Encoding” is added (shown by a plus sign and a small positional-encoding symbol) before entering the stack. Each encoder layer contains a “Multi-Head Attention” sublayer followed by an “Add & Norm” block (a residual connection plus normalization), then a “Feed Forward” sublayer followed by another “Add & Norm.” Arrows and curved skip connections indicate residual connections around the attention and feed-forward sublayers.

On the right is the decoder stack, also repeated “N×.” At the bottom, the decoder takes “Outputs (shifted right)” (previous output tokens shifted by one position) into an “Output Embedding,” adds “Positional Encoding,” and passes the result upward through the repeated decoder layers. Each decoder layer has three main sublayers in order: “Masked Multi-Head Attention” (self-attention that is masked so positions cannot attend to future tokens), then “Multi-Head Attention” (attention over the encoder’s output, i.e., encoder–decoder or cross-attention), then a “Feed Forward” network, with an “Add & Norm” block after each of these sublayers. At the top of the decoder stack, a “Linear” layer feeds into “Softmax” to produce “Output Probabilities.” The overall flow shows embeddings plus positional encodings feeding encoder and decoder stacks, with the decoder attending both to past outputs (masked self-attention) and to encoder representations (cross-attention) before generating token probabilities.



wise fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm( x +Sublayer( x )) , where Sublayer( x ) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d model = 512 .

Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i .

## 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

## 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension d k , and values of dimension d v . We compute the dot products of the



--- PAGE 4 ---

## Scaled Dot-Product Attention

![Figure](image_png_images/sample-picture-2.png)

The figure is a small block diagram of a scaled dot‑product attention computation, drawn as a vertical stack of rectangular operation blocks with arrows showing data flow. At the bottom, three inputs are labeled Q, K, and V (query, key, and value). Q and K feed into a first block labeled “MatMul” (matrix multiplication), whose output flows upward into a block labeled “Scale” (scaling the dot products). Above that is a block labeled “Mask (opt.)”, indicating an optional masking step applied to the scaled scores. The masked (or unmasked) scores then go into a block labeled “SoftMax” (softmax normalization). The softmax output feeds into a top “MatMul” block, which also receives V via a separate arrow coming from the right side, indicating multiplication of the attention weights by the value matrix. A long vertical arrow on the right connects from V upward toward the top MatMul, emphasizing that V bypasses the earlier steps and is only used in the final matrix multiplication.



Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

![Figure](image_png_images/sample-picture-3.png)

The figure is a schematic diagram of the multi-head attention mechanism used in transformer models. At the bottom, three separate inputs labeled Q, K, and V (query, key, and value) each flow upward into their own “Linear” projection blocks, indicating that Q, K, and V are first transformed by learned linear layers. The outputs of these linear layers feed into multiple parallel attention heads, depicted as repeated stacked blocks labeled “Scaled Dot-Product Attention,” with a bracket on the right labeled h indicating there are h heads operating in parallel. The outputs from all heads are then combined in a “Concat” (concatenation) block, and the concatenated result is passed through a final “Linear” layer at the top to produce the multi-head attention output. The diagram emphasizes the pipeline: linear projections of Q/K/V, parallel scaled dot-product attention across h heads, concatenation of head outputs, and a final linear transformation.



query with all keys, divide each by √ d k , and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q . The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:


![Formula](formula_images/sample-formula-1.png)

LaTeX: \text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
This equation defines the (scaled dot-product) attention operation: it forms attention weights by taking the dot products between queries \(Q\) and keys \(K\), scaling by \(1/\sqrt{d_k}\), applying a softmax to normalize the scores, and then using the resulting weights to take a weighted combination of the values \(V\). Here \(K^{T}\) is the transpose of \(K\), and \(d_k\) is the key dimensionality used for scaling.




The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √ d k . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of d k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d k [3]. We suspect that for large values of d k , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4 . To counteract this effect, we scale the dot products by 1 √ d k .

## 3.2.2 Multi-Head Attention

Instead of performing a single attention function with d model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d k , d k and d v dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d v -dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

4 To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1 . Then their dot product, q · k = ∑ d k i =1 q i k i , has mean 0 and variance d k .



--- PAGE 5 ---

![Formula](formula_images/sample-formula-2.png)

LaTeX: \mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^{O}\quad\text{where }\mathrm{head}_i=\mathrm{Attention}(QW_i^{Q},KW_i^{K},VW_i^{V})

This defines multi-head attention: it computes h separate attention “heads” by applying an Attention function to linearly projected versions of Q, K, and V using per-head projection matrices \(W_i^{Q}, W_i^{K}, W_i^{V}\), then concatenates the resulting \(\mathrm{head}_i\) outputs and applies an output projection \(W^{O}\).




Where the projections are parameter matrices W Q i ∈ R d model × d k , W i K ∈ R d model × d k , W V i ∈ R d model × d v and W O ∈ R hd v × d model .

In this work we employ h = 8 parallel attention layers, or heads. For each of these we use d k = d v = d model /h = 64 . Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

## 3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [31, 2, 8].
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to -∞ ) all values in the input of the softmax which correspond to illegal connections. See Figure 2.

## 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.


![Formula](formula_images/sample-formula-3.png)

LaTeX: FFN(x)=\max(0, xW_{1}+b_{1})\,W_{2}+b_{2}
This defines a two-layer feed-forward network (FFN) that takes an input \(x\), applies an affine transform \(xW_{1}+b_{1}\), passes it through a ReLU nonlinearity \(\max(0,\cdot)\), then applies a second affine transform with parameters \(W_{2}\) and \(b_{2}\) to produce the output. Here \(W_{1},W_{2}\) are weight matrices (or vectors) and \(b_{1},b_{2}\) are bias terms.




While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is d model = 512 , and the inner-layer has dimensionality d ff = 2048 .

## 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d model. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [24]. In the embedding layers, we multiply those weights by √ d model.

## 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the



--- PAGE 6 ---

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.

![Table](table_images/sample-table-2.png)

The table compares four neural network layer types by their asymptotic computational cost per layer, how many operations must be performed sequentially, and the maximum length of the dependency path between positions (a proxy for how directly information can flow across distant tokens). The columns are “Layer Type,” “Complexity per Layer,” “Sequential Operations,” and “Maximum Path Length.” The symbols use big‑O notation. Here, n denotes the sequence length (number of positions/tokens), d denotes the representation/hidden dimensionality, k denotes convolution kernel width (filter size), and r denotes a restriction factor for a constrained self‑attention variant; log is the logarithm.

For standard self‑attention, the per‑layer compute scales as O(n²·d), reflecting pairwise interactions across all n positions, while sequential operations are O(1), meaning the layer can be computed in parallel across positions. Its maximum path length is O(1), indicating any position can attend to any other in a single layer, giving the shortest possible dependency path.

For recurrent layers, the per‑layer compute is O(n·d²), and sequential operations are O(n), reflecting that the computation proceeds step‑by‑step through the sequence and cannot be fully parallelized across time. The maximum path length is also O(n), meaning information may need to traverse the entire sequence through n steps/layers of recurrence to connect distant positions, yielding the longest dependency paths among the listed options.

For convolutional layers, the per‑layer compute is O(k·n·d²), scaling linearly with sequence length n and kernel width k (and quadratically with d as written), with sequential operations O(1) because convolutions can be applied in parallel over positions. The maximum path length is O(log_k(n)), indicating that stacking convolutions can connect distant positions with a number of layers that grows logarithmically in n when the receptive field expands multiplicatively with kernel size k; larger k reduces the log base effect and shortens the path.

For restricted self‑attention, the per‑layer compute is O(r·n·d), reducing the quadratic n² factor by limiting attention to a subset of positions determined by r, while maintaining O(1) sequential operations (parallelizable like standard attention). Its maximum path length is O(n/r), showing that restricting attention increases the number of hops needed for distant interactions compared with full self‑attention, but can still be shorter than recurrent path lengths when r is sufficiently large.

Overall, the table highlights a tradeoff: full self‑attention achieves constant path length and full parallelism but has the highest dependence on sequence length (quadratic in n), recurrence has linear dependence on n but the worst sequential dependence and longest paths, convolution sits between with parallelism and logarithmic path length but cost proportional to kernel width k, and restricted attention reduces compute relative to full attention while lengthening the dependency path in proportion to n divided by the restriction factor r.



bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d model as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [8].

In this work, we use sine and cosine functions of different frequencies:


![Formula](formula_images/sample-formula-4.png)

LaTeX: \bigcap^{2 i / d_{\pi}}
The expression shows an intersection operator \(\bigcap\) with an upper limit/superscript \(2 i / d_{\pi}\). Here \(i\) and \(d_{\pi}\) appear as symbols in the bound.





![Formula](formula_images/sample-formula-5.png)

LaTeX: \(PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)\)

This defines sinusoidal positional encodings \(PE\) as a function of position \(pos\) and embedding dimension index: even dimensions \(2i\) use a sine and odd dimensions \(2i+1\) use a cosine, with the argument scaled by \(10000^{2i/d_{model}}\) where \(d_{model}\) is the model/embedding dimensionality.




where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2 π to 10000 · 2 π . We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k , PE pos + k can be represented as a linear function of PE pos .

We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

## 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations ( x 1 , ..., x n ) to another sequence of equal length ( z 1 , ..., z n ) , with x i , z i ∈ R d , such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O ( n ) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d , which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [31] and byte-pair [25] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in



--- PAGE 7 ---

the input sequence centered around the respective output position. This would increase the maximum path length to O ( n/r ) . We plan to investigate this approach further in future work.

A single convolutional layer with kernel width k &lt; n does not connect all pairs of input and output positions. Doing so requires a stack of O ( n/k ) convolutional layers in the case of contiguous kernels, or O ( log k ( n )) in the case of dilated convolutions [15], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k . Separable convolutions [6], however, decrease the complexity considerably, to O ( k · n · d + n · d 2 ) . Even with k = n , however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

## 5 Training

This section describes the training regime for our models.

## 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [31]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

## 5.2 Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

## 5.3 Optimizer

We used the Adam optimizer [17] with β 1 = 0 . 9 , β 2 = 0 . 98 and glyph[epsilon1] = 10 -9 . We varied the learning rate over the course of training, according to the formula:


![Formula](formula_images/sample-formula-6.png)

LaTeX: \(\text{lrate}=d_{\text{model}}^{-0.5}\cdot \min(\text{step\_num}^{-0.5},\ \text{step\_num}\cdot \text{warmup\_steps}^{-1.5})\)

This defines a learning-rate schedule \(\text{lrate}\) that scales with the model dimension \(d_{\text{model}}\) and is set to the smaller of two terms: an inverse square-root decay in the training step number \(\text{step\_num}\), and a linear warmup term \(\text{step\_num}\cdot \text{warmup\_steps}^{-1.5}\) that increases during early steps (controlled by \(\text{warmup\_steps}\)).




This corresponds to increasing the learning rate linearly for the first warmup \_ steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup \_ steps = 4000 .

## 5.4 Regularization

We employ three types of regularization during training:

Residual Dropout We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of P drop = 0 . 1 .



--- PAGE 8 ---

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

![Table](table_images/sample-table-3.png)

The table compares several neural machine translation models on two English-to-European-language tasks, English→German (EN-DE) and English→French (EN-FR), reporting translation quality as BLEU score and, for many entries, an estimate of training compute cost measured in floating-point operations (FLOPs). Higher BLEU indicates better translation quality, while higher FLOPs indicates more training compute.

The “Model” column lists individual systems and some ensemble variants, with bracketed numbers indicating cited references. Quality is split into two BLEU columns: “EN-DE” and “EN-FR.” Compute is also split by task into “Training Cost (FLOPs)” for EN-DE and EN-FR, expressed in scientific notation (for example, 3.3·10^18 FLOPs). Not every model has a compute estimate shown.

Across non-ensemble single models, BLEU on EN-DE ranges from 23.75 for ByteNet up to 28.4 for Transformer (big), while BLEU on EN-FR ranges from 38.1 for the Transformer base model up to 41.29 for ConvS2S Ensemble. The Transformer base model scores 27.3 BLEU on EN-DE and 38.1 on EN-FR; scaling to Transformer (big) improves both tasks to 28.4 (EN-DE) and 41.0 (EN-FR), making it the best single-model result on EN-DE and very competitive on EN-FR. Earlier systems show lower EN-DE BLEU in the mid‑20s (e.g., GNMT + RL at 24.6; ConvS2S at 25.16; MoE at 26.03) but stronger EN-FR BLEU around 40–41 (39.92 for GNMT + RL; 40.46 for ConvS2S; 40.56 for MoE). Deep-Att + PosUnk reports only EN-FR BLEU (39.2) without an EN-DE value.

Ensembles generally improve BLEU relative to their corresponding single models, especially on EN-FR. For example, GNMT + RL increases from 39.92 to 41.16 BLEU on EN-FR when ensembled, and from 24.6 to 26.30 on EN-DE. ConvS2S increases from 40.46 to 41.29 on EN-FR and from 25.16 to 26.36 on EN-DE with an ensemble, with 41.29 being the highest EN-FR BLEU in the table. Deep-Att + PosUnk Ensemble reports 40.4 BLEU on EN-FR, higher than the non-ensemble Deep-Att + PosUnk at 39.2.

Training cost values, where present, span roughly 10^18 to 10^21 FLOPs. The lowest listed compute is for the Transformer base on EN-DE at 3.3·10^18 FLOPs, while the highest is for GNMT + RL Ensemble on EN-FR at 1.1·10^21 FLOPs (ConvS2S Ensemble on EN-FR is similarly high at 1.2·10^21). In many cases EN-FR training cost is higher than EN-DE for the same model (e.g., GNMT + RL: 2.3·10^19 vs 1.4·10^20; ConvS2S: 9.6·10^18 vs 1.5·10^20; MoE: 2.0·10^19 vs 1.2·10^20). Ensembles dramatically increase compute compared to non-ensembles (e.g., GNMT + RL from 2.3·10^19 to 1.8·10^20 on EN-DE, and from 1.4·10^20 to 1.1·10^21 on EN-FR). The Transformer (big) shows higher compute than Transformer base on EN-DE (2.3·10^19 vs 3.3·10^18) alongside higher BLEU, illustrating a quality–compute tradeoff; notably, it achieves top or near-top BLEU without approaching the extreme 10^21 FLOPs costs shown for some ensembles.



Label Smoothing During training, we employed label smoothing of value glyph[epsilon1] ls = 0 . 1 [30]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

## 6 Results

## 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2 . 0 BLEU, establishing a new state-of-the-art BLEU score of 28 . 4 . The configuration of this model is listed in the bottom line of Table 3. Training took 3 . 5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41 . 0 , outperforming all of the previously published single models, at less than 1 / 4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate P drop = 0 . 1 , instead of 0 . 3 .

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0 . 6 [31]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50 , but terminate early when possible [31].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU 5 .

## 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

5 We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.



--- PAGE 9 ---

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

![Table](table_images/sample-table-4.png)

The table summarizes an ablation study of a Transformer-style sequence-to-sequence model, reporting how architectural and training hyperparameters affect translation quality and model size. Each row block varies one set of settings relative to a stated “base” configuration, and the rightmost columns report performance on a development set plus parameter count. The main metrics are PPL (dev), meaning development-set perplexity where lower is better; BLEU (dev), a translation accuracy score where higher is better; and params ×10^6, the number of model parameters in millions. Training budget is shown as “train steps” (e.g., 100K or 300K).

The base model uses N=6 layers, d_model=512 (the model/embedding width), d_ff=2048 (the inner feed-forward width), h=8 attention heads, and per-head dimensions d_k=64 and d_v=64 (key and value dimensionality per head). It applies dropout P_drop=0.1 and label smoothing ε_ls=0.1, trains for 100K steps, and achieves PPL 4.92 and BLEU 25.8 with about 65M parameters. This base serves as the reference point for the variations.

Block (A) varies the number of attention heads h while keeping the overall model width fixed, which implicitly changes per-head dimensions (d_k and d_v) to maintain the same total width. The tested head counts include 1, 4, 16, and 32, with corresponding per-head sizes (for example, h=1 uses d_k=d_v=512; h=4 uses 128; h=16 uses 32; h=32 uses 16). Across these settings, BLEU stays in a narrow band around the mid‑25s (roughly 24.9 to 25.8), and PPL stays around ~4.9 to ~5.3, indicating that performance is fairly robust to head count in this range. The strongest BLEU in this block matches the base (25.8) at h=16 with PPL 4.91, while very low head count (h=1) degrades somewhat (PPL 5.29, BLEU 24.9).

Block (B) varies only the per-head key/value dimensionality (d_k and d_v) while keeping other core settings similar, showing results for d_k=d_v of 16 and 32. These runs also report parameter counts around 58–60M (slightly smaller than the base 65M). The smaller per-head dimension (16) yields worse perplexity and BLEU (PPL 5.16, BLEU 25.1) than the 32 setting (PPL 5.01, BLEU 25.4), suggesting that overly small key/value subspaces hurt.

Block (C) explores depth and width trade-offs. One sub-block varies the number of layers N (shown as 2, 4, and 8), with parameter counts increasing with depth (about 36M at 2 layers, 50M at 4 layers, and 80M at 8 layers). Performance improves with more layers: the shallow 2-layer model is clearly worse (PPL 6.11, BLEU 23.7), while 4 layers recovers much of the base performance (PPL 5.19, BLEU 25.3), and 8 layers slightly surpasses it on BLEU (PPL 4.88, BLEU 25.5). Another sub-block varies model width d_model (256 vs 1024), with corresponding per-head dimensions indicated (e.g., at d_model=256, d_k=d_v=32; at d_model=1024, d_k=d_v=128). The smaller width (256) greatly reduces parameters (about 28M) but hurts metrics (PPL 5.75, BLEU 24.5). The larger width (1024) increases parameters substantially (about 168M) and produces better results (PPL 4.66, BLEU 26.0), showing a clear scaling benefit. A further sub-block varies the feed-forward dimension d_ff (1024 vs 4096), with parameters rising as d_ff increases (about 53M at 1024 and 90M at 4096). The larger feed-forward width improves BLEU (26.2 vs 25.4) and slightly improves PPL (4.75 vs 5.12), indicating that expanding the feed-forward network is beneficial.

Block (D) varies regularization hyperparameters: dropout probability P_drop and label smoothing ε_ls. The table shows runs with P_drop set to 0.0 and 0.2, and ε_ls set to 0.0 and 0.2. With dropout removed (P_drop=0.0), perplexity worsens (PPL 5.77) and BLEU drops (24.6), implying overfitting or poorer generalization. Increasing dropout to 0.2 yields PPL 4.95 and BLEU 25.5, close to the base. For label smoothing, removing it (ε_ls=0.0) gives PPL 4.67 and BLEU 25.3, while stronger smoothing (ε_ls=0.2) gives PPL 5.47 but the best BLEU in this block (25.7), illustrating that perplexity and BLEU do not always move together and that heavier smoothing can slightly improve BLEU despite worse PPL.

Block (E) changes the positional encoding method, replacing sinusoidal positional signals with learned positional embeddings. This variant reports PPL 4.92 and BLEU 25.7, essentially matching the base perplexity and slightly improving BLEU, suggesting positional embeddings are a viable alternative without major impact.

The final “big” model scales up multiple dimensions at once: N=6, d_model=1024, d_ff=4096, h=16, with dropout 0.3 and 300K training steps. It achieves the best reported development metrics in the table (PPL 4.33 and BLEU 26.4) but at a much larger size of about 213M parameters, highlighting a strong accuracy gain from increased capacity plus longer training and heavier dropout.



In Table 3 rows (B), we observe that reducing the attention key size d k hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [8], and observe nearly identical results to the base model.

## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor .

Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.



--- PAGE 10 ---

## References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016.
- [2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR , abs/1409.0473, 2014.
- [3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR , abs/1703.03906, 2017.
- [4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733 , 2016.
- [5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR , abs/1406.1078, 2014.
- [6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357 , 2016.
- [7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR , abs/1412.3555, 2014.
- [8] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2 , 2017.
- [9] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850 , 2013.
- [10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 770-778, 2016.
- [11] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
- [12] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation , 9(8):1735-1780, 1997.
- [13] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410 , 2016.
- [14] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR) , 2016.
- [15] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2 , 2017.
- [16] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations , 2017.
- [17] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR , 2015.
- [18] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722 , 2017.
- [19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130 , 2017.
- [20] Samy Bengio Łukasz Kaiser. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS) , 2016.



--- PAGE 11 ---

- [21] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025 , 2015.
- [22] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing , 2016.
- [23] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304 , 2017.
- [24] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859 , 2016.
- [25] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909 , 2015.
- [26] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 , 2017.
- [27] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research , 15(1):1929-1958, 2014.
- [28] Sainbayar Sukhbaatar, arthur szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28 , pages 2440-2448. Curran Associates, Inc., 2015.
- [29] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems , pages 3104-3112, 2014.
- [30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR , abs/1512.00567, 2015.
- [31] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144 , 2016.
- [32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR , abs/1606.04199, 2016.
