function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% 		Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% first get the unregularized cost and gradient.
[J, grad] = costFunction(theta, X, y);

% now calculate the regularization term.
theta_2n = theta(2:end);
reg_term = theta_2n' * theta_2n * lambda / (2 * m);

% now we can get the regularized cost calculated from the unregularized cost and the regularization term.
J = J + reg_term;

% and we can regularize the gradient.
grad(2:end) = grad(2:end) + lambda * theta(2:end) / m;

end
