# test94_attributes.R
x <- 1:5

attr(x, "description") <- "small integer vector"
print(attributes(x))
print(attr(x, "description"))

dim(x) <- c(5, 1)
print(x)
print(attributes(x))
