function ret = maxInput()
  ret = 86;
end

function formatted = formatInput(x)
  formatted = (vec(x))(1:maxInput())';
end

function input = loadInput(file)
  input = formatInput(load(file), maxInput());         
end

# format noise as (maxInput x n) matrix
function formatted = formatNoiseInnput()
  x = load("noise.dat");
  x = vec(x);
  num_col = length(x) / maxInput();
  num_col = floor(num_col);
  x = x(1:num_col * maxInput);
  formatted = reshape(x, num_col, maxInput());
end

# Split data set to test, training and cross validation
function [train, cv, test] = split_data_set(A)
  m = size(A, 1);
  num_train = floor(m * 0.6);        # 60%
  num_cv = floor(m * 0.2);           # 20%
  num_test = m - num_train - num_cv; # 20%
  train = A([1:num_train],:);
  cv = A([num_train + 1:num_train + num_cv],:);
  test = A([num_train + num_cv + 1:num_train + num_cv + num_test],:);              
end

function [X_train, y_train, X_cv, y_cv, X_test, y_test] = loadNoiseData()
  noise = formatNoiseInnput();
  [X_train, X_cv, X_test] = split_data_set(noise);

  class_index = 1;
  y_train = createY(X_train, class_index);
  y_cv = createY(X_cv, class_index);
  y_test = createY(X_test, class_index);
end

function [X_train, y_train, X_cv, y_cv, X_test, y_test] = loadWarrior2Data()
  A = loadInput("warrior2_right1.dat");
  for i = 2:10
    A = vertcat(A, loadInput(sprintf("warrior2_right%d.dat", i)));
  end
  [X_train, X_cv, X_test] = split_data_set(A);

  class_index = 3;
  y_train = createY(X_train, class_index);
  y_cv = createY(X_cv, class_index);
  y_test = createY(X_test, class_index);
end

# Create Matrix Y, which is [0 1 0] * m where 1 = feature we classfies
function y = createY(X, index)
  num_labels = 3;
  y = zeros(size(X, 1), num_labels);
  one = ones(size(X, 1), 1);
  y(:,index) = one;
end

function [X_train, y_train, X_cv, y_cv, X_test, y_test] = loadDownwardDogData()
  A = loadInput("downdog1.dat");
  for i = 2:10
    A = vertcat(A, loadInput(sprintf("downdog%d.dat", i)));
  end
  [X_train, X_cv, X_test] = split_data_set(A);
  class_index = 2;
  y_train = createY(X_train, class_index);
  y_cv = createY(X_cv, class_index);
  y_test = createY(X_test, class_index);
end

function [X_train, y_train, X_cv, y_cv, X_test, y_test] = loadAllData()
  [X1_train, y1_train, X1_cv, y1_cv, X1_test, y1_test] = loadNoiseData();
  [X2_train, y2_train, X2_cv, y2_cv, X2_test, y2_test] = loadDownwardDogData();
  [X3_train, y3_train, X3_cv, y3_cv, X3_test, y3_test] = loadWarrior2Data();
  X_train = vertcat(X1_train, X2_train, X3_train);
  y_train = vertcat(y1_train, y2_train, y3_train);
  X_cv = vertcat(X1_cv, X2_cv, X3_cv);
  y_cv = vertcat(y1_cv, y2_cv, y3_cv);
  X_test = vertcat(X1_test, X2_test, X3_test);
  y_test = vertcat(y1_test, y2_test, y3_test);
end

input_layer_size  = maxInput();
hidden_layer_size = 200;
num_labels = 3;

fprintf("==== Start ====\n");

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

[X_train, y_train, X_cv, y_cv, X_test, y_test] = loadAllData();

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


y_cv
pred = predict(Theta1, Theta2, X_cv);

y_test;
pred = predict(Theta1, Theta2, X_test);
