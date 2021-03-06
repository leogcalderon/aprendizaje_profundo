# 6. Optimization

## 6.1 Optimization and deep learning
In this chapter, we are going to focus specifically on the performance of the optimization algorithm in minimizing the objective function. In deep learning, most objective functions are complicated and do not have analytical solutions. Instead, we must use numerical optimization algorithms.

**Challenges:**
#### Local Minima:

![lm](imgs/local_minima.png)

#### Saddle Points:

![sp](imgs/saddle.png)

Eigenvalues of the function’s Hessian matrix at the zero-gradient position  are
1. (+) -> Local minimum
2. (-) -> Local maximum
3. (+) and (-) -> Saddle point

Convex functions are those where the eigenvalues of the Hessian are never negative. Sadly, though, most deep learning problems do not fall into this category.

#### Vanishing gradients:

For larger values of x in tanh(x), the gradient is close to nill. Optimization will get stuck for a long time before we make progress. This turns out to be one of the reasons that training deep learning models was quite tricky prior to the introduction of the ReLU activation function.

### 6.2 Convexity:
Even though the optimization problems in deep learning are generally nonconvex, they often exhibit some properties of convex ones near local minima.

**Sets:**  a set is convex if for any a,b in set, the line connecting a and b is also in the set. If X and Y are convex, the union is also convex.

![c](imgs/convex.png)

**Convex functions:** The requiremnt that X is a convex set is necessary.
![fc](imgs/fconvex.png)

## 6.3 Gradient Descent
### 6.3.1 One dimension

x ← (x − ηf′(x))

The value of function f(x) might decline. Therefore, in gradient descent we first choose an initial value x and a constant η>0 and then use them to continuously iterate x until the stop condition is reached, for example, when the magnitude of the gradient |f′(x)| is small enough or the number of iterations has reached a certain value.

#### Learning rate
If we use a learning rate that is too small, it will cause x to update very slowly, requiring more iterations to get a better solution.

#### Local Minima
Depending on our choice of learning rate and depending on how well conditioned the problem is, we may end up with one of many solutions.

### 6.3.2 Multivariate Gradient Descent
Let us consider the situation where x ∈ Rd. That is, the objective function f : Rd → R maps vectors into scalars. Its gradient is a vector consisting of d partial derivatives.

## 6.4 Stochastic Gradient Descent
Stochastic gradient descent (SGD) reduces computational cost at each iteration. At each iteration of stochastic gradient descent, we uniformly sample an index for data examples at random, and compute the gradient to update x.

As we can see, the trajectory of the variables in the SGD is much more noisy than the one we observed in gradient descent. This is due to the stochastic nature of the gradient.

**Dynamic Learning Rate:** Replacing η with a time-dependent learning rate η(t).

![dlr](imgs/dlr.png)

## 6.5 Minibatch Stochastic Gradient Descent
Gradient Descent is not particularly data efficient whenever data is very similar. Stochastic Gradient Descent is not particularly computationally efficient since CPUs and GPUs cannot exploit the full power of vectorization

We can increase the computational efficiency of this operation by applying it to a minibatch of observations at a time. That is, we replace the gradient over a single observation by one over a small batch

Now we can compare the time vs. loss for the previous four experiments. As can be seen, although SGD converges faster than GD in terms of number of examples processed, it uses more time to reach the same loss than GD because computing the gradient example by example is not as efficient. Minibatch SGD is able to trade-off convergence speed and computation efficiency. A minibatch size of 10 is more efficient than SGD; a minibatch size of 100 even outperforms GD in terms of runtime.

![](imgs/time.png)

## 6.6 Momentum

![m](imgs/momentum.png)

This effectively replaces the instantaneous gradient by one that’s been averaged over multiple past gradients, v is called momentum. It accumulates past gradients similar to how a heavy ball rolling down the objective function landscape integrates over past forces. Large β amounts to a long-range average, whereas small β amounts to only a slight correction relative to a gradient method (it solves Ill conditioned problems).

## 6.7 Adagrad

Parameters associated with infrequent features only receive meaningful updates whenever these features occur. Given a decreasing learning rate we might end up in a situation where the learning rate either decreases too slowly for frequent features or too quickly for infrequent ones.

A possible hack to redress this issue would be to count the number of times we see a particular feature and to use this as a clock for adjusting learning rates.

AdaGrad addresses this by replacing the rather crude counter s(i,t)
by an aggregate of the squares of previously observed gradients.

![](imgs/adagrad.png)

Due to the cumulative effect of st, the learning rate continuously decays.

## 6.8 RMSProp

One of the key issues in AdaGrad is that the learning rate decreases at a predefined schedule. Adagrad accumulates the squares of the gradient into a state vector. As a result st keeps on growing without bound due to the lack of normalization, essentially linearly as the algorithm converges.

![rmsprop](imgs/rmsprop.png)

## 6.9 Adadelta

Adadelta has no learning rate parameter. Instead, it uses the rate of change in the parameters itself to adapt the learning rate. It requires two state variables to store the second moments of gradient and the change in parameters.

Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients.

![ad](imgs/adadelta.png)

## 6.10 Adam
Adam combines features of many optimization algorithms into a fairly robust update rule.

It uses exponential weighted moving averages to obtain an estimate of both the momentum and also the second momento of the gradient.

![](imgs/adam1.png)

Correspondingly the normalized state variables are given by:

![](imgs/adam2.png)

First, we rescale the gradient: ![](imgs/adam3.png)

Then we update:
![](imgs/adam4.png)

### 6.10.1 Yogi

One of the problems of Adam is that it can fail to converge even in convex settings when the second moment estimate in st blows up.

For gradients with significant variance we may encounter issues with convergence. They can be amended by using larger minibatches or by switching to an improved estimate for st. Yogi offers such an alternative.

## 6.11 Learning rate scheduling

Decreasing the learning rate during training can lead to improved accuracy and (most perplexingly) reduced overfitting of the model.

* A **piecewise** decrease of the learning rate whenever progress has plateaued is effective in practice. Essentially this ensures that we converge efficiently to a suitable solution and only then reduce the inherent variance of the parameters by reducing the learning rate.

* **Cosine** schedulers are popular for some computer vision problems.

* A **warmup** period before optimization can prevent divergence.
