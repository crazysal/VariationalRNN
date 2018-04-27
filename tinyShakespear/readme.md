## Variational Recurrent Network (VRNN)

Implementation based on Chung's *A Recurrent Latent Variable Model for Sequential Data* [arXiv:1506.02216v6].

### 1. Network design

There are three types of layers: input (x), hidden(h) and latent(z). We can compare VRNN sided by side with RNN to see how it works in generation phase.

- RNN: $h_o + x_o -> h_1 + x_1 -> h_2 + x_2 -> ...$
- VRNN: with $ h_o \left\{
\begin{array}{ll}
      h_o -> z_1 \\
      z_1 + h_o -> x_1\\
      z_1 + x_1 + h_o -> h_1 \\
\end{array} 
\right .$ 
with $ h_1 \left\{
\begin{array}{ll}
      h_1 -> z_2 \\
      z_2 + h_1 -> x_2\\
      z_2 + x_2 + h_1 -> h_2 \\
\end{array} 
\right .$

It is clearer to see how it works in the code blocks below. This loop is used to generate new text when the network is properly trained. x is wanted output, h is deterministic hidden state, and z is latent state (stochastic hidden state). Both h and z are changing with repect to time.

### 2. Training

The VRNN above contains three components, a latent layer genreator $h_o -> z_1$, a decoder net to get $x_1$, and a recurrent net to get $h_1$ for the next cycle.

The training objective is to make sure $x_0$ is realistic. To do that, an encoder layer is added to transform $x_1 + h_0 -> z_1$. Then the decoder should transform $z_1 + h_o -> x_1$ correctly. This implies a cross-entropy loss in the "tiny shakespear" or MSE in image reconstruction.

Another loose end is  $h_o -> z_1$. Statistically, $x_1 + h_0 -> z_1$ should be the same as $h_o -> z_1$, if $x_1$ is sampled randomly. This constraint is formularize as a KL divergence between the two.

>#### KL Divergence of Multivariate Normal Distribution
>![](https://wikimedia.org/api/rest_v1/media/math/render/svg/8dad333d8c5fc46358036ced5ab8e5d22bae708c)

Now putting everything together for one training cycle.

$\left\{
\begin{array}{ll}
      h_o -> z_{1,prior} \\
      x_1 + h_o -> z_{1,infer}\\
      z_1 <- sampling N(z_{1,infer})\\
      z_1 + h_o -> x_{1,reconstruct}\\
      z_1 + x_1 + h_o -> h_1 \\
\end{array} 
\right . $
=>
$
\left\{
\begin{array}{ll}
      loss\_latent = DL(z_{1,infer} | z_{1,prior}) \\
      loss\_reconstruct = x_1 - x_{1,reconstruct} \\
\end{array} 
\right .
$
