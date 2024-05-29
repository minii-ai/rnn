# rnn

Time traveling to the past to implement a simple character-level RNN. We'll write the forward pass, backpropagation through time (BPTT), and gradient descent algorithm from scratch without the help of Pytorch. Yes, you'll have to manually calculate the gradients. It will be worth it. After this, we'll train it on small text datasets like _Shakespeare_ and _How to Get Rich_ by Naval.

# Installation

The only packages we need are `numpy` and `tqdm`.

```bash
poetry init
poetry shell
poetry install
```

# Dataset

To build a dataset, make a txt file. Every sequence will be seperated by a newline.

Snippet from `./data/stevejobs.txt`. In the example below, there are 2 items.

```txt
"Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart."
― Steve Jobs

"Being the richest man in the cemetery doesn't matter to me. Going to bed at night saying we've done something wonderful... that's what matters to me."
― Steve Jobs

...
```

# Quick Start

Train and sample from RNN.

## Training

Training on Steve Jobs dataset for 100000 iterations at `lr = 1e-1` with sequence length `s`. Weights will be saved to `./weights.pkl`. Checkpoints every 1000 steps, sample beginning with char `c` at temperature `0.5`.

```bash
python3 train.py \
    -d "./data/stevejobs.txt" \
    -i 100000 \
    -lr 1e-1 \
    -s 25 \
    -sp "./weights.pkl" \
    -vs 1000 \
    -vc c \
    -vt 0.5 \
```

## Inference

Load weights and generate `3` characters starting with `C`.

```bash
python3 inference.py \
    -w ./weights.pkl \
    -c C \
    -n 3 \
```

# Background

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/02/image-80.png)

_For those interested in how the forward and backward pass works._

RNNs are a type of neural net that operate on sequences. They use a hidden state vector and an input vector to predict an output vector (ie. probabilities over tokens) and the next hidden state. The next hidden state is used to predict the next token, so on and so forth. The recurrence relationship, where the next hidden state depends on the previous hidden state gives them the name **Recurrent Neural Nets**.

## Forward Pass

RNNs have a simple API. They take in a hidden state vector and an input vector to produce an output vector and the next hidden state.

**Weights and Bias Matrices**
$$W_{xh}\in ({n \times m}), \ W_{hh} \in ({n \times n}), b_h \in ({1 \times n}) $$
$$W_{hy} \in ({l \times n}), b_y \in ({1 \times l})$$

**Input Vector**
$$x_t \in ({1 \times m})$$

**Hidden State Vector**
$$h_t \in ({1 \times n})$$

**Foward Pass**

We have all the ingredients for the forward pass! Our choice of activation is `tanh` and `softmax`. Tanh squeezes the activations between -1 and 1 and softmax gives us output probabilities.

```math
z_t^h = x_tW_{xh}^{T} + h_{t-1}W_{hh}^{T} + b_h
```

```math
h_t = \tanh(a_t)
```

```math
z_t^y = h_tW_{hy}^{T} + b_y
```

```math
\hat{y}_t = \text{softmax}(b_t)
```

## Backward Pass

After the forward pass, we'll compute the loss and gradients of the weight and bias matrices. Then do gradient descent.

### Loss

We use cross entropy loss between the predicted token probability and the target token across all time steps.

```math
L = \sum_{t=1}^{T}L_t(\hat{y}_t, y_t) = \sum_{t=1}^{T}{\sum_{i = 1}^{C}-y_t^i
\log(\hat{y}_t^i)}
```

### Backpropagation Through Time

Source: https://phillipi.github.io/6.882/2020/notes/6.036_notes.pdf

The hardest part is keeping track of matrix shapes. Do multiply the shapes of the partials to check if the shapes make sense. Don't memorize. You can derive everything from first principles just by following the RNN section of the textbook above. Phillip Isola does an amazing job at explaining how to implement BPTT and where the gradients come from. 

Hopefully you'll come to the same result! My derivation is slightly different because I use row vectors instead of column vectors $x_t \in (1 \times m)$.

Just peel back each layer and apply the chain rule.


#### Gradient of Loss w.r.t $z_t^y$

https://cs231n.github.io/neural-networks-case-study/#grad

$$
\frac{\partial L_t}{\partial z_t^y} = \hat{y_t} - y_t
$$

$$(1 \times l) = (1 \times l) - (1 \times l)$$

#### Gradient of Tanh
$$\frac{\partial \tanh(u)}{\partial u} = 1 - \tanh(u)^2$$

#### Gradient of 1st Layer

Gradient of the $W_{hy}$

$$
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial L_t}{\partial z_t^y}\frac{\partial z_t^y}{\partial W_{hy}} = \sum_{t=1}^T (\frac{\partial L_t}{\partial z_t^y})^T h_t
$$

$$
(l \times n) = (l \times 1) \times (1 \times n)
$$

Gradient of the $b_y$

$$
\frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial L_t}{\partial z_t^y}\frac{\partial z_t^y}{\partial b_y} = \frac{\partial L_t}{\partial z_t^y}
$$

$$
(1 \times l) = (1 \times l)
$$


#### Gradients of 2nd Layer

This is the most tricky layer. Let's define some terms which will be useful.

#### Definitions

We'll be taking gradients of the future loss with respect to a hidden state. The future loss is defined as follows. Note the recurrence in the definition (we can write $F_{t-1}$ in terms of $F_t$).

$$F_t = \sum_{u = t + 1}^TL_u$$

$$F_{t - 1} = L_t + \sum_{u = t + 1}^TL_u = L_t + F_t$$

#### Gradient of Future Loss w.r.t hidden state

$$
\delta^{h_{t-1}} = \frac{\partial F_{t-1}}{\partial h_{t-1}} 
= \frac{\partial}{\partial h_{t-1}} {\sum_{u = t}^TL_u} 
= \frac{\partial}{\partial h_t} {\sum_{u = t}^TL_u} \frac{\partial h_t}{\partial h_{t-1}}
= (\frac{\partial L_t}{\partial h_t} + \frac{\partial}{\partial h_t}\sum_{u = t + 1}^TL_u) \frac{\partial h_t}{\partial h_{t-1}}
= (\frac{\partial L_t}{\partial h_t} + \delta^{h_t}) \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial h_{t-1}}
$$

$$
(1 \times n)
$$

Note that $\frac{\partial L_t}{\partial h_t} + \delta^{h_t} = \frac{\partial F_{t-1}}{\partial h_t}$.


Also note that $\frac{\partial F_T}{\partial h_T} = 0 \in (1 \times n)$ because there $L_{T+1} ...$ do not exist. $L_T$ is the loss at the final timestep, so the final hidden state $h_T$ will have no effect on future losses, hence the zero gradient. 

#### Gradient of Loss w.r.t hidden state
$$
\frac{\partial L_t}{\partial h_t} = \frac{\partial L_t}{\partial z_t^y}\frac{\partial z_t^y}{\partial h_t} = \frac{\partial L_t}{\partial z_t^y} W_{hy}
$$

$$
(1 \times n) = (1 \times l) \times (l \times n)
$$

#### Gradient of hidden state w.r.t to its input before activation $z_t^h$

$$\frac{\partial h_t}{\partial z_t^h} = 1 - h_t^2$$

$$(1 \times n)$$

Really this is a $(n \times n)$ diagonal matix but b/c $\frac{\partial h_t^i}{\partial z_t^j} = 0$ when $i \neq j$, I decided to grab diagonal and stuff it into a $(1 \times n)$ row vector. The reason being activation are applied element-wise.


#### Gradient of $F_{t-1}$ w.r.t hidden state

Everything is going to come together nicely now. This is just the sum of 2 gradients we defined above.

$$
\frac{\partial F_{t-1}}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial}{\partial h_t} \sum_{u = t + 1}^TL_u = \frac{\partial L_t}{\partial h_t} + \delta^{h_t}
$$

$$
(1 \times n) = (1 \times n) + (1 \times n)
$$


#### Gradient of $F_{t-1}$ w.r.t $z_t^h$

$$
\frac{\partial F_{t-1}}{\partial z_t^h} = \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial z_t^h}
$$

$$
(1 \times n) = (1 \times n) \times (1 \times n)
$$


#### Gradient of hidden state weight matrices

Let's calculate them now!

$$
\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial z_t^h} \frac{\partial z_t^h}{\partial W_{xh}} = \sum_{t=1}^T (\frac{\partial F_{t-1}}{\partial z_t^h})^T x_t
$$

$$
(n \times m) = (n \times 1) \times (1 \times m)
$$

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial z_t^h} \frac{\partial z_t^h}{\partial W_{xh}} = \sum_{t=1}^T (\frac{\partial F_{t-1}}{\partial z_t^h})^T h_{t-1}
$$


$$
(n \times n) = (n \times 1) \times (1 \times n)
$$

#### Gradient of hidden state bias vector
$$
\frac{\partial L}{\partial b_{h}} = \sum_{t=1}^{T} \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial z_t^h} \frac{\partial z_t^h}{\partial b_h} = \sum_{t=1}^T \frac{\partial F_{t-1}}{\partial z_t^h}
$$

$$
(1 \times n) = (1 \times n)
$$

#### Computing $\delta^h_{t-1}$
At timestep $t$ to compute $\frac{\partial F_{t-1}}{\partial h_t}$, we need $\delta^h_t$. For timestep $t-1$, we will need to compute $\delta^h_{t-1}$. Let's revisit the definition of this gradient.

$$
\delta^{h_{t-1}} = \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial h_{t-1}}
$$

Let's apply the chain rule to $\frac{\partial h_t}{\partial h_{t-1}}$. 

$$
\delta^{h_{t-1}} = \frac{\partial F_{t-1}}{\partial h_t} \frac{\partial h_t}{\partial z_t^h}\frac{\partial z_t^h}{\partial h_{t-1}}
$$

Simplify.

$$
\delta^{h_{t-1}} = \frac{\partial F_{t-1}}{\partial z_t^h} \frac{\partial z_t^h}{\partial h_{t-1}} = \frac{\partial F_{t-1}}{\partial z_t^h} W_{hh}
$$

$$(1 \times n) = (1 \times n) \times (n \times n)$$

We now have everything we need to implement backprop. In `rnn.py` we will translate all of this to code.

### Gradient Descent

We'll implement a version of gradient descent called Adagrad. One of the common problems training RNNs are the exploding vanishing gradients. Adagrad *adapts* our learning rate so that we take smaller steps when the gradients are big and bigger steps when gradients are small. This improves our training stability significantly. In fact, using vanilla stochastic gradient descent, training does not converge.

```math
g_t = \nabla_{\theta} L(\theta_t)
```
```math
G_t = G_{t-1} + g_t^2
```
```math
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
```

We implement this in `train.py`.

## Resources
Would not have made it without these. Read them.

- https://phillipi.github.io/6.882/2020/notes/6.036_notes.pdf
- https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
- https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- https://gist.github.com/karpathy/d4dee566867f8291f086
- https://cs231n.github.io/neural-networks-case-study/#grad
- https://explained.ai/matrix-calculus/
- https://www.youtube.com/watch?v=0XdPIqi0qpg
