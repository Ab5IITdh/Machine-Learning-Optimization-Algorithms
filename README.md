# Machine Learning Optimization Algorithms

## Introduction

Optimization algorithms are the backbone of machine learning models, enabling them to learn from data by minimizing error functions. These algorithms adjust model parameters iteratively to reduce the difference between predicted and actual values. This repository explores three fundamental optimization algorithms:

- **Gradient Descent (GD)**
- **Stochastic Gradient Descent (SGD)**
- **Adam Optimizer**

Each of these algorithms has unique characteristics, advantages, and limitations. We will cover their mathematical foundations, implementation details, and provide visualizations for better understanding.

---

## 1. Gradient Descent (GD)

Gradient descent is an optimization algorithm that can help us optimize our loss function more efficiently than the "manual" approach we tried above. As the name suggests, we are going to leverage the **gradient** of our loss function to help us optimize our model parameters. The gradient is just a vector of (partial) derivatives of the loss function w.r.t the model parameters. Sounds complicated, but it's not at all (as I'll hopefully show you).

In plain English, the gradient will tell us two things:

1. **Which direction** to move our parameter in to decrease loss (i.e., should we increase or decrease its value?)
2. **How far** to move it (i.e., should we adjust it by 0.1 or 2 or 50 etc.?)

> If you need a refresher on gradients, check out [Appendix A: Gradients Review](appendixA_gradients.ipynb).
### Mathematical Formulation

Gradient Descent is an iterative optimization algorithm used to minimize a cost function \( J(\theta) \) by updating the model parameters in the direction of the negative gradient. The update rule is given by:

```math
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)
```

where:
- $$\( \theta \)$$ represents the model parameters,
- $$\( \alpha \)$$ is the learning rate,
- $$\( \nabla J(\theta_t) \)$$ is the gradient of the cost function with respect to the parameters.

### Working Mechanism

1. Compute the gradient of the cost function.
2. Update parameters in the opposite direction of the gradient.
3. Repeat until convergence.
4. 
## Gradient Descent With One Parameter

Let's forget about the intercept now and just work with this simple linear model:  
\[
\hat{y_i} = w x_i
\]
For this model, the loss function has the form:

\[
f(w) = \frac{1}{n} \sum^{n}_{i=1} ((w x_i) - y_i)^2
\]

The gradient of this function with respect to the parameter \( w \) is:

\[
\frac{d}{dw} f(w) = \frac{1}{n} \sum^{n}_{i=1} 2x_i (w x_i - y_i)
\]
```python
Let's code that up and calculate the gradient of our loss function at a slope of \( w = 0.5 \):

def gradient_descent(x, y, w=0.5, alpha=0.00001, ε=1e-8, max_iterations=10000, print_progress=1000):
    iterations = 0
    dw = float('inf')
    while abs(dw) > ε and iterations <= max_iterations:
        g = gradient(x, y, w)  # calculate current gradient
        dw = alpha * float(g)  # ensure g is converted to a float
        w -= dw
        iterations += 1
        if iterations % print_progress == 0:
            print(f"Iteration {iterations}. w = {w:.4f}, dw = {dw:.4f}")
    print("Terminated!")
    print(f"Iteration {iterations - 1}. w = {w:.4f}.")
    return w
```
### Effect Of Learning Rate : 
```python
plot_gradient_descent(x, y, w=0.5, alpha=0.00001)
```
![newplot (1)](https://github.com/user-attachments/assets/b44084f1-2670-4f19-b225-3564317e58ad) \\

```python
plot_gradient_descent(x, y, w=0.5, alpha=0.00005)
```
![newplot (2)](https://github.com/user-attachments/assets/82644b59-e957-4261-ac34-311bc02344b3)

```python
plot_gradient_descent(x, y, w=0.5, alpha=0.00015)
```
![newplot (3)](https://github.com/user-attachments/assets/53ab5f3d-9d0e-458f-871e-1f99cf936190)

```python
plot_gradient_descent(x, y, w=0.5, alpha=0.00018, max_iterations=4)
```
![newplot (4)](https://github.com/user-attachments/assets/f8a12873-5308-4c3f-a4a5-8d9739a530a4)

## Gradient Descent With Two Parameter

Most of the models you'll be working with will have more than just one parameter to update - neural networks typically have hundreds, thousands, and even millions of parameters! So, let's extend the above workflow to two parameters, the intercept ($w_0$) and the slope ($w_1$). Just to help you get a visual of what's going on, let's take our "manual grid search approach" from earlier and make a plot of it but this time with two parameters:

Above is the surface of MSE for different values of `intercept` ($w_0$) and `slope` ($w_1$). The approach for implementing gradient descent is exactly as before, but we're operating on two parameters now and the gradient of the intercept is a little different to the slope:

$$f(w)=\frac{1}{n}\sum^{n}_{i=1}((w_0 + w_1x)-y_i))^2$$

$$\frac{\partial{}}{\partial{}w_0}f(w)=\frac{1}{n}\sum^{n}_{i=1}2((w_0 + w_1x) - y_i)$$

$$\frac{\partial{}}{\partial{}w_1}f(w)=\frac{1}{n}\sum^{n}_{i=1}2x_i((w_0 + w_1x) - y_i)$$

```python
def gradient_descent(x, y, w, alpha, ϵ=2e-4, max_iterations=5000, print_progress=10):
    """Gradient descent for optimizing simple linear regression."""
    
    print(f"Iteration 0. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
    iterations = 1  # init iterations
    dw = np.array(2 * ϵ)      # init. dw
    
    while abs(dw.sum()) > ϵ and iterations <= max_iterations:
        g = gradient(x, y, w)  # calculate current gradient
        dw = alpha * g         # change in w
        w -= dw                # adjust w based on gradient * learning rate
        if iterations % print_progress == 0:  # periodically print progress
            print(f"Iteration {iterations}. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")
        iterations += 1        # increase iteration
        
    print("Terminated!")
    print(f"Iteration {iterations - 1}. Intercept {w[0]:.2f}. Slope {w[1]:.2f}.")

gradient_descent(x, y, w=[10, 0.5], alpha=0.00001)
```
![newplot (5)](https://github.com/user-attachments/assets/5eedbab2-fb75-4c1c-b56f-b412f3eeebe5)



### Pros & Cons

✅ Converges smoothly with a well-chosen learning rate.  
❌ Computationally expensive for large datasets.

### Visualization

![newplot](https://github.com/user-attachments/assets/8556db9b-433a-4e3b-a19f-6ed39c5efbaf) \\

#### It looks like a slope of 0.9 gives us the lowest MSE (~184.4). But you can imagine that this "grid search" approach quickly becomes computationally intractable as the size of our data set and number of model parameters increases
---

## 2. Stochastic Gradient Descent (SGD)

### Mathematical Formulation

Unlike batch Gradient Descent, SGD updates model parameters using only a single training example at each iteration:

```math
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t; x_i, y_i)
```

where \( (x_i, y_i) \) is a randomly selected training sample.

### Working Mechanism

1. Shuffle dataset.
2. Pick a random sample and compute its gradient.
3. Update parameters using this single sample.
4. Repeat for all samples.

### Pros & Cons

✅ Faster updates, leading to quicker convergence.  
✅ Works well for large datasets.  
❌ Highly noisy updates, leading to fluctuating convergence.

### Visualization

*(Add a plot showing noisy but faster convergence compared to GD.)*

---

## 3. Adam Optimizer (Adaptive Moment Estimation)

### Mathematical Formulation

Adam combines the benefits of Momentum and RMSProp. It maintains exponentially weighted moving averages of past gradients and squared gradients:

```math
\begin{aligned}
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla J(\theta_t) \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) (\nabla J(\theta_t))^2 \\
    \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
    \hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
    \theta_{t+1} &= \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
```

where:
- \( m_t \) and \( v_t \) are the first and second moment estimates,
- \( \beta_1 \) and \( \beta_2 \) are decay rates,
- \( \epsilon \) is a small constant for numerical stability.

### Working Mechanism

1. Compute biased estimates of gradient moments.
2. Correct bias.
3. Perform adaptive parameter updates.

### Pros & Cons

✅ Faster convergence compared to GD and SGD.  
✅ Handles sparse gradients effectively.  
❌ Computationally heavier due to moment calculations.

### Visualization

*(Add a plot comparing Adam with GD and SGD.)*

---

## Conclusion

Each optimization algorithm has its strengths and weaknesses. Choosing the right optimizer depends on the dataset and model complexity. This repository provides code implementations and visual demonstrations to facilitate better understanding.

Stay tuned for more updates and examples!

---

## Repository Structure

```
├── README.md (This document)
├── gradient_descent.py (Implementation of GD)
├── stochastic_gradient_descent.py (Implementation of SGD)
├── adam_optimizer.py (Implementation of Adam)
├── visualizations/ (Contains plots)
└── datasets/ (Sample datasets for testing)
```

## Contributing

Contributions are welcome! If you find any errors or want to add improvements, feel free to submit a pull request.

## References

1. D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." arXiv:1412.6980, 2014.  
2. I. Goodfellow, Y. Bengio, A. Courville. "Deep Learning." MIT Press, 2016.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

