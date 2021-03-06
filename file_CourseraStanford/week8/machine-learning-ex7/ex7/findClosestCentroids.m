function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for X_i=1:size(X, 1)
      % dat khoang cach nho nhat bang duong vo cung
      khoangcach_min=Inf;
      % gan chi so bang Inf
      idex_min=Inf;
      for cent_i=1:K
            khoangcach_cur=sum((X(X_i,:)-centroids(cent_i,:)).^2); 
            if khoangcach_min>khoangcach_cur
                  khoangcach_min=khoangcach_cur;
                  idex_min=cent_i;
            end
      end
      % cap nhat chi so cho idx
      idx(X_i)=idex_min;
end
% =============================================================

end

