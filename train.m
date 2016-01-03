function formatted = formatInput(x, maxNumInput)
  formatted = (vec(x))(1:maxNumInput);
end

function input = loadInput(file)
  input = formatInput(load(file), 80);         
end

downdog1 = loadInput("downdog1.dat");
size(downdog1)

