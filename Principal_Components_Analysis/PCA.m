function Z = PCA(x, threshold)
% x: original data, matrix
% threshold: threshold for contribution rate, scalar

% standardization
x_mean = mean(x, 1);
std_x = std(x, 0, 1);
X = (x - x_mean) ./ std_x;

% covariance
R = cov(X);

% eigenvalue and eigenvector
[A, eig_value] = eig(R);
[eig_value, idx] = sort(diag(eig_value), "descend");
A = A(:,idx);

% contribution rate
sum_eig_value = sum(eig_value);
ctb_rate = cumsum(eig_value / sum_eig_value);
taken_num = find(ctb_rate >= threshold, 1);

% calculate principal components
Z = X * A;
Z = Z(:, 1:taken_num);
end
