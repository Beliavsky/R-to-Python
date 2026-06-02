x = 1:4
xmat = matrix(x, 2, 2)
print(x)
print(xmat)
xmat = matrix(x, 4, 1)
print(xmat)
xmat = matrix(x, 1, 4)
print(xmat)
xmat = matrix(x, 2, 3) # use recycling
print(xmat)
