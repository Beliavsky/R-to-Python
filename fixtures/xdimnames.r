# test58_dimnames.R
x <- matrix(1:6, nrow = 2, ncol = 3)
rownames(x) <- c("r1", "r2")
colnames(x) <- c("c1", "c2", "c3")

print(x)
print(rownames(x))
print(colnames(x))
print(x["r1", "c2"])
