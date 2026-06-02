# test88_rle.R
x <- c(1, 1, 1, 2, 2, 3, 1, 1)

r <- rle(x)

print(r)
print(inverse.rle(r))
