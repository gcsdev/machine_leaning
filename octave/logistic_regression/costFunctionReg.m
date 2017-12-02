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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions=sigmoid(X*theta); %predictions with new function
toYEqual1=-y'*log(predictions); % when y is equal 1
toYEqual0=(-(1-y)'*log(1-predictions)); %when y is equal 0
shiftTheta=theta(2:size(theta));%using shift to first element
newTheta=[0;shiftTheta]; % create new theta with shifting
J=(1/m)*(toYEqual1+toYEqual0)+(lambda/(2*m))*newTheta'*newTheta;% general cost function plus step for regularization
partialDerivative =(1/m)*(X'*(predictions-y)+lambda*newTheta);%Computing the partial derivative with regularization
grad=partialDerivative;








% =============================================================

end
