# Principal Components Analysis

Essense: Map data points into a lower dimension space.

Rotate the coordinate to find axis with largest variance and take 
first $k$ axis to represent the entire system.

## Steps:
original data $X$

$$X = \begin{pmatrix} 
\vec x_{1} & \vec x_{2} & \cdots & \vec x_{p}
\end{pmatrix}$$

### Target: 
Find new indices 

$$Z = \begin{pmatrix} \vec z_{1} & \vec z_{2} & \cdots & \vec z_{p}\end{pmatrix}$$

satisfying condition of 

$$\sigma_1^2>\sigma_2^2>\cdots>\sigma_p^2$$

where $z_n$ is called the $n^{th}$ principal component.

### step1: standardization
- calculate mean value of each column:
  
$$\bar x_j = \displaystyle\sum_{i=1}^nx_{ij}$$

- calculate standard deviation:
  
$$\sigma_j = \sqrt{\dfrac{\displaystyle\sum_{i=1}^n (x_{ij} - \bar x_{j})}{n-1}}$$

- standardization:
  
$$X_{ij} = \dfrac{x_{ij} - \bar x_{j}}{\sigma_j}$$

### step2: covariance matrix

$$r_{ij} = \dfrac{1}{n-1}\sum_{k=1}^n (X_{ki} - \bar X_i)(X_{kj} - \bar X_j)$$

$$R = \begin{pmatrix}
r_{11} & r_{12} & \cdots & r_{1p}\\
r_{21} & r_{22} & \cdots & r_{2p}\\
\vdots & \vdots & \ddots & \vdots\\
r_{n1} & r_{n2} & \cdots & r_{np}\\
\end{pmatrix}$$

### step3: calculate eigenvalues and eigenvectors of $R$
Suppose $R$ has eigenvalues $\lambda_1>\lambda_2>\cdots>\lambda_p$, corresponding
eigenvectors $\vec a_1, \vec a_2, \cdots, \vec a_p$

$$\vec a_{i} = \begin{pmatrix}a_{1i}
\\
a_{2i}
\\
\vdots
\\
a_{pi}
\end{pmatrix}$$

**Notice: necessary to sort the eigenvalues**

### step4: calculate contribution rate 
- The $i^{th}$ contribution rate: $\alpha_i = \dfrac{\lambda_i}{\sum \lambda_k}$
- Accumulated contribution rate: $\sum G = \dfrac{\sum^i\lambda_k}{\sum^p\lambda_k}$

Meaning of accumulated contribution rate:

|1|2|3|$\cdots$|
|:---:|:---:|:---:|:---:|
|$\alpha_1$|$\alpha_1+\alpha_2$|$\alpha_1+\alpha_2+\alpha_3$|$\cdots$|

Take when $\alpha_1+\alpha_2+\alpha_3>0.8$

### step4: find principal components

$$\vec z_i = \sum_{k=1}^pa_{ki}\vec X_k$$

Larger coefficient of $X_k$ means more information carried by $X_k$.

In fact if define

$$A = \begin{pmatrix} \vec a_{1} & \vec a_{2} & \cdots & \vec a_{p}\end{pmatrix}$$

then $Z = XA$.

After step4, select components with accumulated contribution rate > 80% to analyse.

## Codes in matlab:
```matlab
function Z = pca(x, threshold)
% x: original data, matrix
% threshold: threshold for contribution rate, scalar

% standardization
x_mean = mean(x, 1);
std_x = std(x, 0, 1);
X = (x - x_mean) ./ std_x;

% covariance
R = cov(X);

% eigenvalue and eigenvector
[A, eig_value] = eig(R);
[eig_value, idx] = sort(diag(eig_value), "descend");
A = A(:,idx);

% contribution rate
sum_eig_value = sum(eig_value);
ctb_rate = cumsum(eig_value / sum_eig_value);
taken_num = find(ctb_rate >= threshold, 1);

% calculate principal components
Z = X * A;
Z = Z(:, 1:taken_num);
end
```

test example:
```matlab
t = 0.2:1e-1:(0.8-1e-3);
x = 0.2 * meshgrid(t) + 0.3 * rand(length(t), length(t));
Z = pca(x, 0.8);

plot(x, "b.")
hold on
plot(Z, "r.")
hold off
```
output:
```
t = 1×6    
    0.2000    0.3000    0.4000    0.5000    0.6000    0.7000

x = 6×6    
    0.0676    0.1830    0.2772    0.2417    0.1989    0.1977
    0.2005    0.2868    0.1069    0.2348    0.3791    0.2222
    0.1431    0.2918    0.1988    0.2072    0.1504    0.2867
    0.2330    0.3258    0.1883    0.3996    0.1484    0.2464
    0.2962    0.0754    0.1921    0.3826    0.3984    0.4379
    0.1016    0.3382    0.1982    0.3034    0.4007    0.2349

Z = 6×3    
   -1.6301    1.6685   -0.5253
   -0.0142   -1.8812   -0.6397
   -1.0126    0.2038    0.1161
    0.1833   -0.1446    1.9331
    3.1529    0.7907   -0.4444
   -0.6792   -0.6372   -0.4399


```
