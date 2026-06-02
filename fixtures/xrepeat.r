# test09_repeat_break_next.R
i <- 0
s <- 0

repeat {
  i <- i + 1

  if (i %% 2 == 0) {
    next
  }

  s <- s + i

  if (i >= 9) {
    break
  }
}

print(s)
