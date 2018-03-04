function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% for i = 1:m
%     J += (theta(1,1) +theta(2,1)*X(i,2)-  y(i))^2;
% end

%% disp( X)
%% Implementation Note
%% We store each example as a row in the the X
%% matrix in Octave/MATLAB. To take into account the intercept term (θ 0 ),
%% we add an additional first column to X and set it to all ones. This allows
%% us to treat θ 0 as simply another ‘feature’.
%% Column 1 of X is always 1, helps in coding vectorized form

J = sum(((theta' * X')' - y) .^ 2);


J = J / (2 * m);



% =========================================================================

end
