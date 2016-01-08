function b = isEuclidDistance()
  b = 1;
end

function ret = maxInput()
  # use x^2 + y^2 + z^2
  if isEuclidDistance() == 1
    ret = 85;
  else
    ret = 85 * 3; # num of rows * (x, y, z)
  end
end

function outputThetaAsFile(name, Theta)
  fid = fopen(sprintf("%s.c", name), "w");
  fprintf(fid, "#define NUM_%s_ROW %d\n", toupper(name), size(Theta, 1));
  fprintf(fid, "#define NUM_%s_COL %d\n", toupper(name), size(Theta, 2));
  fprintf(fid, "static double %s[] = {\n", name);
  theta_vec = vec(Theta);
  for i = 1:length(theta_vec)
    fprintf(fid, "%g", theta_vec(i));
    if i != length(theta_vec)
      fprintf(fid, ",");
    end
  end
  fprintf(fid, "};\n");
  fclose(fid);
end

function formatted = formatInput(x)
  if isEuclidDistance() == 1
     x = x .^ 2 + x .^ 2;
     x = sum(x, 2);
     x = x(1:maxInput(),:);
     formatted = x';
  else
    formatted = (vec(x))(1:maxInput())';
  end
end

function input = loadInput(file)
  input = formatInput(load(file), maxInput());         
end

# format noise as (maxInput x n) matrix
function formatted = formatNoiseInnput()
  x = load("noise.dat");
  if isEuclidDistance() == 1
    x = x .^ 2 + x .^ 2;
    x = sum(x, 2);
    x = vec(x);
    num_col = length(x) / maxInput();
    num_col = floor(num_col);
    x = x(1:num_col * maxInput);
    formatted = reshape(x, num_col, maxInput());
  else
    x = vec(x);
    num_col = length(x) / maxInput();
    num_col = floor(num_col);
    x = x(1:num_col * maxInput);
    formatted = reshape(x, num_col, maxInput());
    size(formatted)
  end
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

function [X_train, y_train, X_val, y_val, X_test, y_test] = loadNoiseData()
  noise = formatNoiseInnput();
  [X_train, X_val, X_test] = split_data_set(noise);

  class_index = 1;
  y_train = createY(X_train, class_index);
  y_val = createY(X_val, class_index);
  y_test = createY(X_test, class_index);
end

function [X_train, y_train, X_val, y_val, X_test, y_test] = loadWarrior2Data()
  A = loadInput("warrior2_right1.dat");
  for i = 2:10
    A = vertcat(A, loadInput(sprintf("warrior2_right%d.dat", i)));
  end
  [X_train, X_val, X_test] = split_data_set(A);

  class_index = 3;
  y_train = createY(X_train, class_index);
  y_val = createY(X_val, class_index);
  y_test = createY(X_test, class_index);
end

# Create Matrix Y, which is [0 1 0] * m where 1 = feature we classfies
function y = createY(X, index)
  num_labels = 3;
  y = zeros(size(X, 1), num_labels);
  one = ones(size(X, 1), 1);
  y(:,index) = one;
end

function [X_train, y_train, X_val, y_val, X_test, y_test] = loadDownwardDogData()
  A = loadInput("downdog1.dat");
  for i = 2:10
    A = vertcat(A, loadInput(sprintf("downdog%d.dat", i)));
  end
  [X_train, X_val, X_test] = split_data_set(A);
  class_index = 2;
  y_train = createY(X_train, class_index);
  y_val = createY(X_val, class_index);
  y_test = createY(X_test, class_index);
end

function [X_train, y_train, X_val, y_val, X_test, y_test] = loadAllData()
  [X1_train, y1_train, X1_cv, y1_cv, X1_test, y1_test] = loadNoiseData();
  [X2_train, y2_train, X2_cv, y2_cv, X2_test, y2_test] = loadDownwardDogData();
  [X3_train, y3_train, X3_cv, y3_cv, X3_test, y3_test] = loadWarrior2Data();
  X_train = vertcat(X1_train, X2_train, X3_train);
  y_train = vertcat(y1_train, y2_train, y3_train);
  X_val = vertcat(X1_cv, X2_cv, X3_cv);
  y_val = vertcat(y1_cv, y2_cv, y3_cv);
  X_test = vertcat(X1_test, X2_test, X3_test);
  y_test = vertcat(y1_test, y2_test, y3_test);
end

function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)

  m = size(X, 1);
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);

  for i = 1:m
    xi = X(1:i, :);
    yi = y(1:i, :);

    [nn_params, cost] = trainWithLambdaInternal(xi, yi, lambda);
    [J grad] = nnCostFunction(nn_params, ...
                              inputLayerSize(), ...
                              hiddenLayerSize(), ...
                              numClassificationLabeles(), ...
                              xi, yi, lambda);


    error_train(i) = J;
    [J grad] = nnCostFunction(nn_params, ...
                              inputLayerSize(), ...
                              hiddenLayerSize(), ...
                              numClassificationLabeles(), ...
                              Xval, yval, 0); # lambda should be 0
    error_val(i) = J;
  end;
end;

# Compute errors for some lambda candidates,
# and choose lambda which generate min error
function [error_train, error_val] = runValidationCurve(X_train, y_train, X_val, y_val)

  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);

  for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    [nn_params, cost] = trainWithLambdaInternal(X_train, y_train, lambda);

    [J grad] = nnCostFunction(nn_params, ...
                              inputLayerSize(), ...
                              hiddenLayerSize(), ...
                              numClassificationLabeles(), ...
                              X_train, y_train, lambda);
    error_train(i) = J;
    [J grad] = nnCostFunction(nn_params, ...
                              inputLayerSize(), ...
                              hiddenLayerSize(), ...
                              numClassificationLabeles(), ...
                              X_val, y_val, lambda);
    error_val(i) = J;
  end

  plot(lambda_vec, error_train, lambda_vec, error_val);
  legend('Train', 'Cross Validation');
  xlabel('lambda');
  ylabel('Error');

  fprintf('Drawing lambda/error graph\n');
  pause;

  fprintf('lambda\t\tTrain Error\tValidation Error\n');
  for i = 1:length(lambda_vec)
    fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
  end
end

function [nn_params, cost] = trainWithLambdaInternal(X_train, y_train, lambda)
  initial_Theta1 = randInitializeWeights(inputLayerSize(), hiddenLayerSize());
  initial_Theta2 = randInitializeWeights(hiddenLayerSize(), numClassificationLabeles());
  initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

  options = optimset('MaxIter', 50);
  costFunction = @(p) nnCostFunction(p, ...
                                     inputLayerSize(), ...
                                     hiddenLayerSize(), ...
                                     numClassificationLabeles(), X_train, y_train, lambda);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
end

function [Theta1, Theta2] = trainWithLambda(X_train, y_train, lambda)
  [nn_params, cost] = trainWithLambdaInternal(X_train, y_train, lambda);

  Theta1 = reshape(nn_params(1:hiddenLayerSize() * (inputLayerSize() + 1)), ...
                   hiddenLayerSize(), (inputLayerSize() + 1));
  Theta2 = reshape(nn_params((1 + (hiddenLayerSize() * (inputLayerSize() + 1))):end), ...
                 numClassificationLabeles(), (hiddenLayerSize() + 1));
end

function drawLearningCurve(X_train, y_train, X_val, y_val, lambda)
  m = size(X_train, 1);
  [error_train, error_val] = ...
  learningCurve(X_train, y_train, ...
                X_val, y_val, ...
                lambda);
  plot(1:m, error_train, 1:m, error_val);
  pause;
end

#### Training Configuration ####
function i = inputLayerSize()
  i = maxInput();
end;

function h = hiddenLayerSize()
 h = 150;
end;

function n = numClassificationLabeles()
 n = 3;
end;

fprintf("==== Start ====\n");

[X_train, y_train, X_val, y_val, X_test, y_test] = loadAllData();

#plot(X_val(1,:));
## hold on;
## plot(X_val(39,:), "1");
## plot(X_val(38,:), "2");
## plot(X_val(37,:), "3");
## size(X_val(39,:))
## pause;

# which lambda works best?
#runValidationCurve(X_train, y_train, X_val, y_val);

# We know this value is the best
lambda = 0.001;

# draw learning curve to see how # of training data affect the error
#drawLearningCurve(X_train, y_train, X_val, y_val, lambda);

[Theta1, Theta2] = trainWithLambda(X_train, y_train, lambda);

size(Theta1)
size(Theta2)

# Predict
[dummy, answer] = max(y_val, [], 2);

save "Theta1.dat", Theta1;
save "Theta2.dat", Theta2;
pred = predict(Theta1, Theta2, X_val);

fprintf('answer\t\tPredict\n');
for i = 1:size(answer, 1)
  result = '';
  if pred(i) != answer(i)
     result = '(*)';
  end
  fprintf(' %f\t%f %s\n', ...
          answer(i), pred(i), result);
end

outputThetaAsFile("theta1", Theta1);
outputThetaAsFile("theta2", Theta2);

# validationCurve.m を実装する
# これまとめる
# 制度が悪いものを実際のデータをグラフで見てみる
# x .^2 を追加してみる？



