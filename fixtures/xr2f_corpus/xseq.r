# seq()

seq(1, 5)
# 1 2 3 4 5

seq(5, 1)
# 5 4 3 2 1

seq(0, 1, by = 0.2)
# 0.0 0.2 0.4 0.6 0.8 1.0

seq(1, 10, by = 3)
# 1 4 7 10

seq(1, 10, length.out = 5)
# 1.00 3.25 5.50 7.75 10.00

seq(from = 2, to = 8)
# 2 3 4 5 6 7 8

seq(along.with = c("a", "b", "c"))
# 1 2 3

seq(length.out = 4)
# 1 2 3 4


# seq.int()

seq.int(1, 5)
# 1 2 3 4 5

seq.int(5, 1)
# 5 4 3 2 1

seq.int(2, 10, by = 2)
# 2 4 6 8 10

seq.int(10, 2, by = -2)
# 10 8 6 4 2

seq.int(from = 3, to = 11, length.out = 5)
# 3 5 7 9 11


# seq_along()

x <- c(10, 20, 30)
seq_along(x)
# 1 2 3

seq_along(letters[1:5])
# 1 2 3 4 5

seq_along(list("a", "b", "c"))
# 1 2 3

seq_along(numeric(0))
# integer(0)

for (i in seq_along(x)) print(x[i])


# seq_len()

seq_len(5)
# 1 2 3 4 5

seq_len(1)
# 1

seq_len(0)
# integer(0)

n <- 4
seq_len(n)
# 1 2 3 4

m <- nrow(matrix(1:6, 2, 3))
seq_len(m)
# 1 2


# common safe loop patterns

x <- c(4, 7, 9)
for (i in seq_along(x)) print(i)

n <- length(x)
for (i in seq_len(n)) print(x[i])

print(seq(2,6))
print(seq(2,6,3))
print(seq(2,6,0.5))
