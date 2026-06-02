# test101_simple_naive_bayes.R
train_x <- data.frame(
  color = factor(c("red", "red", "blue", "blue", "red", "blue")),
  size = factor(c("s", "l", "s", "l", "s", "s"))
)

train_y <- factor(c("yes", "yes", "no", "no", "yes", "no"))

predict_one <- function(color, size) {
  classes <- levels(train_y)
  score <- numeric(length(classes))
  names(score) <- classes

  for (cl in classes) {
    idx <- train_y == cl
    prior <- mean(idx)
    p_color <- mean(train_x$color[idx] == color)
    p_size <- mean(train_x$size[idx] == size)
    score[cl] <- prior * p_color * p_size
  }

  return(names(which.max(score)))
}

print(predict_one("red", "s"))
print(predict_one("blue", "l"))
