# test47_uniroot.R
f <- function(x) {
  return(x^3 - 2)
}

ans <- uniroot(f, lower = 0, upper = 2)

print(ans$root)
print(ans$f.root)
print(ans$iter)
