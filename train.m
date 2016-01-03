function formatted = formatInput(x, maxNumInput)
  formatted = (vec(x))(1:maxNumInput)
end

downdog1 = load("downdog1.dat");
size(formatInput(downdog1, 80))

