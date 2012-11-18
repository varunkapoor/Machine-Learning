function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)


m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

% 		following code make predictions using
%       learned neural network. You should set p to a 
%       vector containing labels between 1 to num_labels.



X = [ones(m, 1), X];
a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2, 1), 1), a2];
a3 = a2 * Theta2';

[max_p, p] = max(a3, [], 2);

end
