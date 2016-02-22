function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% tao mot vec to chua cac gia tri cua tham so svm C va sigma
C_vec = [0.01 0.03 0.06 0.1 0.3 0.6 1 3 6 10 30];
sigma_vec = [0.01 0.03 0.06 0.1 0.3 0.6 1 3 6 10 30];
% khoi tao gia tri sai so la vo cung
error_min = Inf;
% khoi tao gia tri C va sigma nho nhat tu 
C_min = Inf;
sigma_min = Inf;
% tim gia tri C va sigma toi uu

for C_i = 1:length(C_vec)
      for sigma_i = 1:length(sigma_vec)
            % gan gia tri cua C bang gia tri cua C trong C_vec
            C_cur=C_vec(C_i);
            % tuong tu vs sigma
            sigma_cur = sigma_vec(sigma_i);
            % huan luyen du lieu
            model = svmTrain(X, y, C_cur, @(x1, x2) gaussianKernel(x1, x2, sigma_cur));
            % du doan 
            predict_result = svmPredict(model, Xval);
            % tinh sai so cua ket qua du doan vs thuc te
            error = mean(double(predict_result ~= yval));
            % neu sai so nho hon sai so mean thi chon cac gia tri tuong ung
            % do
            if error_min > error
                  error_min = error;
                  C_min = C_cur;
                  sigma_min = sigma_cur;
            end
      end
end

C = C_min;
sigma = sigma_min;


% =========================================================================

end
