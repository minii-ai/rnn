# rnn

Time traveling to the past to implement a simple character-level RNN. We'll write the forward pass, backpropagation through time (BPTT), and gradient descent algorithm from scratch without the help of Pytorch. Yes, you'll have to manually calculate the gradients. It will be worth it. After this, we'll train it on small text datasets like _Shakespeare_ and _How to Get Rich_ by Naval.

# Installation

The only packages we need are `numpy` and `tqdm`.

```bash

```

# Quick Start

## Training
```bash

```

## Inference
```bash


```


# Background

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/02/image-80.png)

_For those interested in how the forward and backward pass works._

RNNs are a type of neural net that operate on sequences. They use a hidden state vector and an input vector to predict an output vector (ie. probabilities over tokens) and the next hidden state. The next hidden state is used to predict the next token, so on and so forth. The recurrence relationship, where the next hidden state depends on the previous hidden state gives them the name **Recurrent Neural Nets**.

## Forward Pass

RNNs have a simple API. They take in a hidden state vector and an input vector to produce an output vector and the next hidden state.

**Weights and Bias Matrices**
$$W_{xh}\in \R^{n \times m}, \ W_{hh} \in \R^{n \times n}, b_h \in \R^{1 \times n} $$
$$W_{hy} \in \R^{l \times n}, b_y \in \R^{1 \times l}$$

**Input Vector**
$$x_t \in \R^{1 \times m}$$

**Hidden State Vector**
$$h_t \in \R^{1 \times n}$$

**Foward Pass**

We have all the ingredients for the forward pass! Our choice of activation is `tanh` and `softmax`. Tanh squeezes the activations between -1 and 1 and softmax gives us output probabilities.

$$z_t^h = x_tW_{xh}^{T} + h_{t-1}W_{hh}^{T} + b_h$$

<!-- ```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
``` -->

```math
z_t^h = x_tW_{xh}^{T} + h_{t-1}W_{hh}^{T} + b_h
h_t = \tanh(a_t)
z_t^y = h_tW_{hy}^{T} + b_y
\hat{y}_t = \text{softmax}(b_t)
```

$$h_t = \tanh(a_t) $$
$$z_t^y = h_tW_{hy}^{T} + b_y $$
$$\hat{y}_t = \text{softmax}(b_t)$$

## Backward Pass

After the forward pass, we'll compute the loss and gradients of the weight and bias matrices. Then do gradient descent.

### Loss

We use cross entropy loss between the predicted token probability and the target token across all time steps.

$$L = \sum_{t=1}^{T}L_t(\hat{y}_t, y_t) = \sum_{t=1}^{T}{\sum_{i = 1}^{C}-y_t^i
\log(\hat{y}_t^i)}$$

### Backpropagation Through Time

Source: https://phillipi.github.io/6.882/2020/notes/6.036_notes.pdf

The hardest part is keeping track of matrix shapes. Do multiply the shapes of the partials to check if the shapes make sense. Don't memorize. You can derive everything from first principles just by following the RNN section of the textbook above.

Hopefully you'll come to the same result! My derivation is slightly different because I use row vectors instead of column vectors (ie. $x_t \in \R^{1 \times m}$).

Note $\odot$ is a Hadamard product, it is multiplication applied element-wise. It is also broadcastable so a product between $(1 \times m)$ and $(n \times 1)$ is $(n \times m)$. 

$$
\frac{\partial L_t}{\partial b_t} = \hat{y_t} - y_t \in (1 \times l)
$$

**Useful Gradients to know**
$$\frac{\partial \tanh(u)}{\partial u} = 1 - \tanh(u)^2$$

**Gradient of Tanh**

$$
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial z_t^y}{\partial W_{hy}} \frac{\partial L_t}{\partial z_t^y}^T = \sum_{t=1}^Th_t \odot (\hat{y_t} - y_t)^T \in (l \times n)
$$


$$
\frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial z_t^y}{\partial b_y} \frac{\partial L_t}{\partial z_t^y} = \hat{y_t} - y_t \in (1 \times l)
$$

**Gradients of 2nd Layer**

Let's define some terms which will be useful. This is the most tricky layer.


$F_t = \sum_{u = t + 1}^TL_u$

$F_{t - 1} = L_t + \sum_{u = t + 1}^TL_u$


$$\begin{align*}
\delta^{h_{t-1}} &= \frac{\partial F_{t-1}}{\partial h_{t-1}} \\ &= \frac{\partial}{\partial h_{t-1}} {\sum_{u = t}^TL_u} \\ &= \frac{\partial h_t}{\partial h_{t-1}}\frac{\partial}{\partial h_t} {\sum_{u = t}^TL_u} \\ &= \frac{\partial h_t}{\partial h_{t-1}} (\frac{\partial L_t}{\partial h_t} + \frac{\partial}{\partial h_t}\sum_{u = t + 1}^TL_u) \\ &= \frac{\partial h_t}{\partial h_{t-1}} (\frac{\partial L_t}{\partial h_t} + \delta^{h_t})

\end{align*}$$


$$\frac{\partial h_t}{\partial z_t^h} = 1 - h_t^2 \in (1 \times n)$$
*Really this is a (n x n) diagonal matix but b/c $\frac{\partial h_t^i}{\partial z_t^j} = 0$ when $i \neq j$, I decided to grab diagonal and stuff it into a (1 x n) row vector. The reason being activation are applied element-wise.*





Let's calculate them now!

$$
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial z_t^h}{\partial W_{xh}} \frac{\partial h_t}{\partial z_t^h} \frac{\partial F_{t-1}}{\partial h_t} = \sum_{t=1}^T x_t \odot (1 - h_t^2)
$$

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial z_t^h}{\partial W_{xh}} \frac{\partial h_t}{\partial z_t^h} \frac{\partial F_{t-1}}{\partial h_t} = \sum_{t=1}^T x_t \odot (1 - h_t^2)
$$

$$
\frac{\partial L}{\partial b_{h}} = \sum_{t=1}^{T} \frac{\partial z_t^h}{\partial W_{xh}} \frac{\partial h_t}{\partial z_t^h} \frac{\partial F_{t-1}}{\partial h_t} = \sum_{t=1}^T (1 - h_t^2)
$$


### Gradient Descent

We go in the opposite direction of the gradient to minimize the loss.

## Resources
