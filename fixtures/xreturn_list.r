# test12_function_return_list.R
summary_stats <- function(x) {
  out <- list(
    n = length(x),
    mean = mean(x),
    sd = sd(x),
    min = min(x),
    max = max(x)
  )
  return(out)
}

x <- c(2, 4, 6, 8, 10)
ans <- summary_stats(x)

print(ans)
print(ans$mean)
print(ans[["sd"]])
