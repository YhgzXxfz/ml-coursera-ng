function [error_train error_val] = learningCurveRandomSelection...
    (X, y, Xval, yval, lambda, times)

% Number of training examples
m = size(X, 1);
mv = size(Xval, 1);

% You need to return these values correctly
t = zeros(m, times);
v = zeros(m, times);

for k = 1:times
    for i = 1:m
        indexes = randsample(m, i);
        Xi = X(indexes, :); yi = y(indexes);
        theta = trainLinearReg(Xi,yi,lambda);

        t(i,k) = linearRegCostFunction(Xi,yi,theta,0);
        
        indexes_val = randsample(size(Xval,1), i);
        v(i,k) = linearRegCostFunction(...
            Xval(indexes_val,:),...
            yval(indexes_val),...
            theta,...
            0);
    end
end

error_train = mean(t, 2)';
error_val = mean(v, 2)';

end