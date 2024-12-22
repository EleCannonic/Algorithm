function Z = pca(x, threshold, method)
% x: original data, matrix
% threshold: threshold for contribution rate, scalar
% method: method taken, ED(eigenvalue decomposition) or SVD(singular value
%         decomposition), optional

% default method
if nargin < 3
    method = "SVD";
end

% prevent error on non-square
[r_x, c_x] = size(x);
if r_x ~= c_x
    method = "SVD";
    warning("ED can't work for non-square matrix. Method switched to SVD");
end

% standardization
x_mean = mean(x, 1);
std_x = std(x, 0, 1);
X = (x - x_mean) ./ std_x;

if method == "ED"
    % covariance matrix
    R = cov(X);

    % eigenvalue and eigenvector
    [V, eig_value] = eig(R);
    [eig_value, idx] = sort(diag(eig_value), "descend");
    V = V(:,idx);

elseif method == "SVD"
    % singular value decomposition
    [~, S, V] = svd(X);

    % sort eigenvalues
    eig_value = sqrt(diag(S));
    [eig_value, idx] = sort(eig_value, "descend");

    V = V(:, idx);
end

% contribution rate
sum_eig_value = sum(eig_value);
ctb_rate = cumsum(eig_value / sum_eig_value);
taken_num = find(ctb_rate >= threshold, 1);

% calculate principal components
Z = X * V;
Z = Z(:, 1:taken_num);

end
