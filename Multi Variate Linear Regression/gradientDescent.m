function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%f = fopen('o.txt','w');
for iter = 1:num_iters
	temp1 = theta(1) - (alpha/m) * sum(((X*theta)-y).* X(:,1));
	temp2 = theta(2) - (alpha/m) * sum(((X*theta)-y).* X(:,2));
	theta(1) = temp1;
	theta(2) = temp2;

    % 	Perform a single gradient step on the parameter vector
    %               theta. 
    %
  
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	%fprintf(f,theta,J_history(iter));
	%save data theta;
	%save data J_history(iter);
end
%fclose(f);
end
