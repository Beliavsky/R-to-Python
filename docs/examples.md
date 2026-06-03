# Small translation examples

This page shows small R snippets and representative Python output from
`xr2p.py`. Generated code may change as the translator improves, but these
examples show the intended style: idiomatic Python where possible, with runtime
helpers only where R semantics need them.

## Simple loop and printing

R:

```r
for (i in 1:3) {
  cat(i, i^2, "\n")
}
```

Python:

```python
for i in range(1, 4):
    print(i, i ** 2)
```

## Vector arithmetic

R:

```r
x <- c(1, 2, 3, 4, 5)
y <- x^2 + 10
print(y)
```

Python:

```python
x = r_c(1, 2, 3, 4, 5)
y = x ** 2 + 10
r_s3_print(y)
```

## Monte Carlo calculation

R:

```r
set.seed(123)
n <- 100000
x <- runif(n, min = -1, max = 1)
y <- runif(n, min = -1, max = 1)
inside <- x^2 + y^2 <= 1
pi_hat <- 4 * mean(inside)
cat("estimated pi =", pi_hat, "\n")
```

Python:

```python
import numpy as np

np.random.seed(123)
n = 100000
x = np.random.uniform(-1, 1, size=n)
y = np.random.uniform(-1, 1, size=n)
inside = x ** 2 + y ** 2 <= 1
pi_hat = 4 * np.mean(inside)
print("estimated pi =", pi_hat)
```

## Matrix creation and column-major order

R:

```r
x <- 1:6
m <- matrix(x, nrow = 2, ncol = 3)
print(m)
```

Python:

```python
import numpy as np

x = r_seq(1, 6)
m = np.resize(r_matrix_data(x), 2 * 3).reshape((2, 3), order='F')
r_s3_print(m)
```

R fills matrices by column. The generated NumPy reshape uses `order='F'` to
preserve that behavior.

## Functions and lists

R:

```r
summ <- function(x) {
  list(n = length(x), mean = mean(x), sd = sd(x))
}
ans <- summ(c(2, 4, 6, 8))
print(ans$mean)
```

Python:

```python
def summ(x):
    return RList(n=r_length(x), mean=np.mean(x), sd=np.std(x, ddof=1), _r_names=['n', 'mean', 'sd'])

ans = summ(r_c(2, 4, 6, 8))
r_print(ans.mean, colnames=getattr(ans, 'mean_colnames', None))
```

## Data frames

R:

```r
df <- data.frame(name = c("a", "b", "c"), x = c(1, 2, 3))
print(df[df$x >= 2, ])
```

Python:

```python
import pandas as pd

df = r_data_frame(name=r_c("a", "b", "c"), x=r_c(1, 2, 3))
r_s3_print(r_subset(df, df.x >= 2, slice(None)))
```

Data frames are represented with pandas DataFrames.

## Named vectors

R:

```r
x <- c(a = 10, b = 20, c = 30)
print(x[c("c", "a")])
```

Python:

```python
x = RNamedVector(r_c(10, 20, 30), ['a', 'b', 'c'])
r_s3_print(r_matrix_index_get(x, r_c("c", "a")))
```

Named vectors keep their labels through common subsetting and arithmetic.

## Optimization caveat

R:

```r
fit <- optim(par = par0, fn = negloglik, method = "BFGS")
```

Python:

```python
fit = optim(par=par0, fn=negloglik, method="BFGS", control=None)
```

The Python wrapper uses `scipy.optimize.minimize`. This is convenient, but
objective functions translated literally from R can be slow if they contain
scalar loops and R-style indexing. For important numerical kernels, `xr2p.py`
may emit specialized NumPy or numba fast paths.
