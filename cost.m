function J = cost(X, y, theta)
  m = size(X, 1);
  #disp(m);
  s1 = size(theta, 1);
  #disp(s1);
  K = size(theta, 2);
  #disp(K);
  %disp(size(theta));
  #disp(size(y));
  
  # Forward propagation
  %disp(size(X));
  a1 = [ones(m, 1) X];
  %disp(size(a1));
  %fprintf("%f x %f * %f x %f", size(theta, 1), size(theta, 2), size(a1', 1), size(a1', 2));
  z2 = theta * a1';
  %disp(size(z2));
  pred = sigmoid(z2)';
  disp(size(pred));
  
  # Error calculation (unregularized)
  #disp(size(yMat));
  yMat = eye(K)(y + 1, :);
  J = 0;
  for i = 1:m
    for k = 1:K
      term1 = (-yMat(i, k) .* log(pred(i, k)));
      term2 = ((1 - yMat(i, k)) .* log(1 - pred(i, k)));
      J += term1 - term2;
      #disp(J);
    endfor
  endfor
  J = (J / m);
endfunction