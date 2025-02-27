# Machine-Learning-Optimization-Algorithms
These algorithms adjust model parameters iteratively to reduce the difference between predicted and actual values.
# Machine Learning Optimization Algorithms

## Introduction

Optimization algorithms are the backbone of machine learning models, enabling them to learn from data by minimizing error functions. These algorithms adjust model parameters iteratively to reduce the difference between predicted and actual values. This repository explores three fundamental optimization algorithms:

- **Gradient Descent (GD)**
- **Stochastic Gradient Descent (SGD)**
- **Adam Optimizer**

Each of these algorithms has unique characteristics, advantages, and limitations. We will cover their mathematical foundations, implementation details, and provide visualizations for better understanding.

---

## 1. Gradient Descent (GD)

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

