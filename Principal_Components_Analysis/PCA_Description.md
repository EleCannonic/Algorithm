# Principal Components Analysis

Essense: Map data points into a lower dimension space.

Rotate the coordinate to find axis with largest variance and take 
first $k$ axis to represent the entire system.

PCA has two methods: eigenvalue decomposition (ED) and singular value decomposition (SVD).

## PCA Based on Eigenvalue Decomposition (ED):
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

### Step1: Standardization
- calculate mean value of each column: 
$$\bar x_j = \displaystyle\sum_{i=1}^nx_{ij}$$
- calculate standard deviation: 
$$\sigma_j = \sqrt{\dfrac{\displaystyle\sum_{i=1}^n (x_{ij} - \bar x_{j})}{n-1}}$$
- standardization:
$$X_{ij} = \dfrac{x_{ij} - \bar x_{j}}{\sigma_j}$$

### Step2: Covariance Matrix
$$r_{ij} = \dfrac{1}{n-1}\sum_{k=1}^n (X_{ki} - \bar X_i)(X_{kj} - \bar X_j)$$
$$R = \begin{pmatrix}
r_{11} & r_{12} & \cdots & r_{1p}\\
r_{21} & r_{22} & \cdots & r_{2p}\\
\vdots & \vdots & \ddots & \vdots\\
r_{n1} & r_{n2} & \cdots & r_{np}\\
\end{pmatrix}$$

### Step3: Calculate Eigenvalues and Eigenvectors of $R$
Suppose $R$ has eigenvalues $\lambda_1>\lambda_2>\cdots>\lambda_p$, corresponding
eigenvectors $\vec a_1, \vec a_2, \cdots, \vec a_p$

$$\vec a_{i} = \begin{pmatrix}
a_{1i}
\\
a_{2i}
\\
\vdots
\\
a_{pi} 
\end{pmatrix}$$

**Notice: necessary to sort the eigenvalues**

### Step4: Calculate Contribution Rate 
- The $i^{th}$ contribution rate: $\alpha_i = \dfrac{\lambda_i}{\sum \lambda_k}$
- Accumulated contribution rate: $\sum G = \dfrac{\sum^i\lambda_k}{\sum^p\lambda_k}$

Meaning of accumulated contribution rate:

|1|2|3|$\cdots$|
|:---:|:---:|:---:|:---:|
|$\alpha_1$|$\alpha_1+\alpha_2$|$\alpha_1+\alpha_2+\alpha_3$|$\cdots$|

Take when $\alpha_1+\alpha_2+\alpha_3>0.8$

### Step5: Find Principal Components

$$\vec z_i = \sum_{k=1}^pa_{ki}\vec X_k$$

Larger coefficient of $X_k$ means more information carried by $X_k$.

In fact if define 

$$A = \begin{pmatrix} \vec a_{1} & \vec a_{2} & \cdots & \vec a_{p}\end{pmatrix}$$ 

then $Z = XA$.

After step4, select components with accumulated contribution rate > 80% to analyse.

## PCA Based on Singular Value Decomposition (SVD):

This is another method of PCA, based on the most powerful matrix decomposition method of SVD.

### Step1: Standardization
This step is the same as that of ED method PCA. We do not repeat it.

### Step2: SVD Operation
- Make singular value decomposition $X = U\Sigma V^T$

$\Sigma$ is a diagonal matrix, whose diagonal is the standard deviation in the direction of each eigenvector.
To find most influential components, we need to sort them in the descending order and pick eigenvectors with
the largest eigenvalues (These eigenvalues represents the influence of the corresponding components).

- Sort the eigenvalue $\{\lambda_n\}$ and corresponding eigenvectors $\{\mathrm{col}_n(V)\}$.

Here it's necessary to explain why $V$ is chosen instead of $U$. In the SVD operation of $n\times d$ matrix $X$,
$U$ is a $n\times n$ matrix while $V$ is a $d\times d$ matrix. In fact, SVD describe a process like the figure below:

<img src="https://github.com/user-attachments/assets/2e9a5071-5f72-4a10-b165-0664ac526491" width="200" height="200">
<img src="https://github.com/user-attachments/assets/6a53d141-f941-4768-a5b9-f5a0e8ca1d5e" width="200" height="200">
<img src="https://github.com/user-attachments/assets/6e329930-4987-4466-b9a5-286830f260bf" width="200" height="200">
<img src="https://github.com/user-attachments/assets/f92c60fb-1b78-4dc7-80be-d9533070c170" width="200" height="200">

This is a "rotate->scale->rotate" process. $U$ operates in the sample space, while $V$ operates in the eigen-space.
Hence, columns of $U$ are directions in the **sample space**, representing the projection on the eigen-space. Columns
of $V$ are the directions in the **eigen-space**, which is our desired space. So the principal components are obviously
included in $V$, instead of $U$.

Let's use a specific example to illustrate it more clear. Consider date matrix

$$
X = \begin{pmatrix}
2&3\\
3&3\\
4&5\\
5&5\\
6&7\\
\end{pmatrix}
$$

This data set includes 5 samples and 2 characteristics for each.
```matlab
X = [2, 5; 
     3, 3;
     4, 5;
     5, 5;
     6, 7];
[U, Sigma, V] = svd(X)
```
```
U = 5×5    
   -0.3466    0.8842   -0.2292    0.0941   -0.1916
   -0.2848   -0.2318   -0.2812   -0.6898   -0.5571
   -0.4320    0.0371    0.8859   -0.0779   -0.1453
   -0.4747   -0.3864   -0.2104    0.6838   -0.3369
   -0.6219   -0.1174   -0.1983   -0.2044    0.7200

Sigma = 5×2    
   14.8209         0
         0    1.8280
         0         0
         0         0
         0         0

V = 2×2    
   -0.6329   -0.7742
   -0.7742    0.6329
```
$\Sigma$ has 2 singular values, meaning that there are 2 principal components. 
It's not difficult to discover that we should choose $V$.


### Step3: Calculate Contribution Rate 
This step is the same as **Step4** of ED method PCA. We do not repeat it.

After this step $V$ plays the role of $A$ in the ED method.

## Comparison of ED and SVD
- A most fatal con of ED is that it can only deal with square matrix.
- Since calculating covariance is expensive when the dimension is high, we suggest that
you choose SVD when there are large square matrix.
- When the matrix is small and square, you can use ED.

## Test in matlab:
test example in eigenvalue decomposition (ED) method:
```matlab
t = 0.2:1e-1:(0.8-1e-3);
x = 0.2*meshgrid(t);
x = x + 0.3*rand(length(t), length(t))
tic
Z = pca(x, 0.8, "ED")
toc
```
output:
```
x = 6×6    
    0.1554    0.3073    0.3519    0.2276    0.2996    0.1606
    0.2149    0.3548    0.3439    0.1938    0.2613    0.2359
    0.1155    0.2791    0.3253    0.1484    0.3288    0.2993
    0.1271    0.1632    0.1582    0.1536    0.3300    0.3363
    0.2251    0.2352    0.2583    0.2269    0.3116    0.2623
    0.1196    0.0923    0.0868    0.1283    0.1301    0.3860

Z = 6×2    
   -1.9590    0.1846
   -1.6189   -0.5739
    0.0392    1.4935
    1.3452    0.7089
   -1.1598   -0.9971
    3.3532   -0.8160

Elapsed time is 0.004434 seconds.
```

test example in SVD method:
```matlab
t = 0.2:1e-1:(0.8-1e-3);
x = 0.2*meshgrid(t);
x = x + 0.3*rand(length(t), length(t))
tic
Z = pca(x, 0.8, "SVD")
toc
```

output:
```
x = 6×6    
    0.0558    0.1853    0.2894    0.1098    0.2582    0.1973
    0.2614    0.3549    0.2800    0.2684    0.4145    0.2685
    0.1207    0.1504    0.1334    0.3646    0.1669    0.2846
    0.1669    0.2703    0.1184    0.3008    0.3767    0.1762
    0.2044    0.2599    0.3797    0.1571    0.3134    0.3169
    0.3228    0.2217    0.1313    0.2107    0.2329    0.2079

Z = 6×4    
   -0.7686   -1.6490    1.4424   -0.1720
    2.1614    0.6757   -0.4007   -0.3613
   -2.1953    0.1447   -1.3954   -0.5224
   -0.0138    1.5043    0.9937   -0.7263
    1.2768   -1.5546   -0.7885    0.1798
   -0.4604    0.8790    0.1485    1.6022

Elapsed time is 0.004191 seconds.
```

## Test in Python
```py
import time
import numpy as np
from pca import pca

X = np.array([
    [2.5, 3.1, 1.2, 0.7, 4.5, 3.3],
    [3.5, 4.2, 1.8, 1.1, 5.1, 4.0],
    [2.8, 3.6, 1.5, 0.9, 4.8, 3.7],
    [3.2, 4.0, 1.7, 1.0, 5.0, 3.9],
    [2.9, 3.4, 1.3, 0.8, 4.6, 3.5],
    [3.0, 3.8, 1.6, 0.9, 4.9, 3.8]
])

# ED test
start_time_ed = time.time()

Z_ed = pca(X, threshold = 0.8, method = "ED")

end_time_ed = time.time()

duration_time_ed = end_time_ed - start_time_ed

print(f"ED: {Z_ed}")
print(f"ED running time: {duration_time_ed:.12f} s")

# SVD test
start_time_svd = time.time()

Z_svd = pca(X, threshold = 0.8, method = "SVD")

end_time_svd = time.time()

duration_time_svd = end_time_svd - start_time_svd

print(f"SVD: {Z_ed}")
print(f"SVD running time: {duration_time_svd:.12f} s")
```
output
```
ED: [[ 3.81991318]
 [-3.48484904]
 [ 0.38930715]
 [-2.00373791]
 [ 1.92783747]
 [-0.64847085]]
ED running time: 0.006947994232 s
SVD: [[ 3.81991318]
 [-3.48484904]
 [ 0.38930715]
 [-2.00373791]
 [ 1.92783747]
 [-0.64847085]]
SVD running time: 0.001019001007 s
```
