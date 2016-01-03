function ret = maxInput()
  ret = 90;
end

function formatted = formatInput(x)
  formatted = (vec(x))(1:maxInput());
end

function input = loadInput(file)
  input = formatInput(load(file), maxInput());         
end

# format noise as (maxInput x n) matrix
function formatted = formatNoiseInnput()
  x = load("noise.dat");
  x = vec(x);
  length(x) / maxInput();
  num_col = length(x) / maxInput();
  num_col = floor(num_col);
  x = x(1:num_col * maxInput);
  formatted = reshape(x, maxInput, num_col);
end

noise = formatNoiseInnput();
size(noise)
downdog1 = loadInput("downdog1.dat");
size(downdog1)

