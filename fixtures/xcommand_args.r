# test116_command_args_like_function.R
main <- function(args) {
  print(args)

  nums <- as.numeric(args)
  print(nums)
  print(sum(nums))
}

main(c("1", "2", "3.5"))
