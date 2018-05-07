function S = calcSimByDist(X, Y, distType)
mx = size(X, 1);
my = size(Y, 1);
S = zeros(mx, my);

if distType == 1 % L1
    for i = 1 : mx
        vector = X(i, :);
        S(i, :) = sum(abs(repmat(vector, my, 1) - Y), 2);
    end
    S = -S;
elseif distType == 2 % L2
    S = pdist2(X, Y, 'euclidean');
    S = -S;
elseif distType == 3 % cosine
    S = pdist2(X, Y, 'cosine');
    S = -S;
elseif distType == 4 % correlation
    S = pdist2(X, Y, 'correlation');
    S = -S;
elseif distType == 5 % dot product
    S = X * Y'; 
elseif distType == 6 % hist intersect
    for i = 1 : mx
        vector = X(i, :);
        temp = min(repmat(vector, my, 1), Y);
        S(i, :) = sum(temp, 2);
    end
elseif distType == 7 % Ki square
    for i = 1 : mx
        A = repmat(X(i, :), my, 1);
        B = Y;
        temp = (A - B).^2 ./ (A + B + eps);
        S(i, :) = sum(temp, 2);
    end
elseif distType == 8 % KL divergence
    for i = 1 : mx
        A = repmat(X(i, :), my, 1);
        B = Y;
        temp = A .* log(A ./ B);
        S(i, :) = -sum(temp, 2);
    end
end