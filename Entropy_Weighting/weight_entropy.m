function w = weight_entropy(X, indextype)
% data: m×n matrix, m observations with n indices
% indextype: a 1×n logical vector, 1 for positive and 0 for negative
% w: 1×n vector, weight of n indices

[m, n] = size(X);
Z = zeros(m, n);
p = zeros(m, n);
e = zeros(1, n);
w = zeros(1, n);

for j = 1:n
    % standardization
    if indextype(j)
        Z(:,j) = (X(:,j) - min(X(:,j))) / (max(X(:,j)) - min(X(:,j)));
    else
        Z(:,j) = (max(X(:,j)) - X(:,j)) / (max(X(:,j)) - min(X(:,j)));
    end
    
    % probability
    eps = 1e-8; % to prevent log(0)
    p(:,j) = Z(:,j) ./ sum(Z(:,j));
    
    % entropy
    e(j) = -(1/log(n)) .* sum(p(:,j) .* log(p(:,j) + eps));
end

% weight
w = (1 - e) / sum(1 - e);

end
