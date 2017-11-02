function J = cost(X, y, theta)
  m = size(X, 1);
  #disp(m);
  s1 = size(theta, 2);
  #disp(s1);
  K = size(theta, 1);
  #disp(K);
  %disp(size(theta));
  #disp(size(y));
  
  #disp(sum((y == 0)(:)));
  
  # Forward propagation
  %disp(size(X));
  #disp(X);
  a1 = [ones(m, 1) X];
  disp(max(a1));
  #disp(X(1, :));
  #disp(a1);
  #disp(sum((a1 == 0)(:)));
  %disp(size(a1));
  %fprintf("%f x %f * %f x %f", size(theta, 1), size(theta, 2), size(a1', 1), size(a1', 2));
  # TODO: For some reason the values of z2 are an order of magnitude larger than they were in the assignment code.
  # This is causing almost every value to become 1 when passed through sigmoid.
  z2 = theta * a1';
  #disp(z2);
  #disp(max(z2(:)));
  #disp(max(theta(:)));
  #disp(sum((z2 == 0)(:)));
  %disp(size(z2));
  #disp(sum((pred == 1)(:)));
  pred = sigmoid(z2)';
  #disp(max(pred(:)));
  
  #disp((z2')(1:5, :));
  #disp(pred(1:5, :));
  
  # Error calculation (unregularized)
  #disp(size(yMat));
  yMat = eye(K)(y, :);
  #disp(yMat);
  J = 0;
  for i = 1:m
    for k = 1:K
      term1 = (-yMat(i, k) .* log(pred(i, k)));
      #disp(term1);
      #disp(1 - yMat(i, k));
      #disp(1 - pred(i, k));
      term2 = ((1 - yMat(i, k)) .* log(1 - pred(i, k)));
      #disp(term2);
      J += term1 - term2;
      #disp(J);
    endfor
  endfor
  J = (J / m);
endfunction