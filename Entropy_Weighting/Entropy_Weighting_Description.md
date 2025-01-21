# Entropy Weighting

The value of information entropy describes the degree of dispersion of data.

entropy ⬇, dispersion ⬆

Data with more dispersion carries more information, which deserves higher weight.

## Steps:

### Step1: Standardization
original data matrix $X$, rows for observation, columns for different indices

Difference between indices:
- Positive Index: The larger the better
- Negative Index: The smaller the better

Standardization:

- For positive indices:

$$z_{ij} = \dfrac{x_{ij} - \min\{\mathrm{col}_j X\}}{\max\{\mathrm{col}_j X\} - \min\{\mathrm{col}_j X\}}$$

- For negative indices:

$$z_{ij} = \dfrac{\max\{\mathrm{col}_ j X\} - x_{ij}}{\max\{\mathrm{col}_j X\} - \min\{\mathrm{col}_j X\}}$$

All indices must be normalized to region $[0, 1]$.

### Step2: Calculate Probability Matrix

$$
p_{ij} = \dfrac{z_{ij}}{\displaystyle\sum_{i=1}^n z_{ij}}
$$

(index / column sum, $\sum_{i=1}^n x_{ij} =$ `sum(X(:,j))` )

### Step3: Entropy Weight

information entropy:

$$
\begin{align*}
e_j &= -\dfrac{1}{\ln n}\sum_{i=1}^n p_{ij}\ln(p_{ij})\\
W_j &= \dfrac{1 - e_j}{\displaystyle\sum_{j=1}^n (1 - e_j)} 
\end{align*}
$$

$W_j$ represents the weight of the $j$th index.

## Test in matlab
```matlab
clear;
X = [1 2 3;
     4 6 8;
     2 8 6;
     3 6 9]
log = ones(1, 3)

w = weight_entropy(X, log)
```

output:
```
X = 4×3    
     1     2     3
     4     6     8
     2     8     6
     3     6     9

log = 1×3    
     1     1     1

w = 1×3    
    0.6035    0.1358    0.2607

```

