# test86_sweep_scale.R
x <- matrix(1:12, nrow = 3, ncol = 4)

m <- colMeans(x)
s <- apply(x, 2, sd)

print(sweep(x, 2, m, "-"))
print(sweep(x, 2, s, "/"))
print(scale(x))
