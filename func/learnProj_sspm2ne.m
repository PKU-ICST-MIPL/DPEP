% Similarity-Similarity Product Maximization
% 1. extend on sspm_ori with consideration of all pairs in the same category and different categories in cross-media types
% 2. do norm to balance the infulence of similarity constraints and dissimilarity constraints
% 3. combine with entity information to set the weights to exploit the fine-grained correlation at the entity level

function [P_T, P_I] = learnProj_sspm2ne(Y, docEntity, trTxtCat, trImgCat, T_tr, I_tr, iterMaxNum, lambda1, lambda2)
lastObjValue = 0;

% deal with feature
% T_tr = T_tr(:, sum(T_tr) ~= 0);
% I_tr = I_tr(:, sum(I_tr) ~= 0);

txtNum = size(T_tr, 1);
imgNum = size(I_tr, 1);
txtDim = size(T_tr, 2);
imgDim = size(I_tr, 2);
labelDim = size(Y, 2);

% Initialize 1: random
rng('default');
P_T = rand(txtDim, labelDim);
P_I = rand(imgDim, labelDim);

% Initialize 2: single-media solution
% P_T = (lambda * eye(txtDim) + T_tr' * T_tr) * T_tr' * Y;
% P_I = (lambda * eye(imgDim) + I_tr' * I_tr) * I_tr' * Y;

% Initialize 3: CFA
% not adapt to Wiki entity
% [U, S, V] = svd(T_tr' * I_tr);
% P_T = U(:, 1:labelDim);
% P_I = V(:, 1:labelDim);


% construct similarity matrix (1, 0, -1)
simMatrix = zeros(txtNum, txtNum);

txtSimMatrix = double(repmat(trTxtCat, 1, txtNum) == repmat(trTxtCat, 1, txtNum)');
imgSimMatrix = double(repmat(trImgCat, 1, imgNum) == repmat(trImgCat, 1, imgNum)');
txtSimMatrix0 = txtSimMatrix - 1;
imgSimMatrix0 = imgSimMatrix - 1;

% set weights by entity information
W = ones(txtNum, imgNum);
% for i = 1 : txtNum
%     sampleEntity = docEntity(i, :);
%     temp = sum(docEntity, 2) + sum(sampleEntity) - sum(repmat(sampleEntity, txtNum, 1) .* docEntity, 2);
%     
%     W(i, :) = sum(repmat(sampleEntity, txtNum, 1) .* docEntity, 2) ./ temp;
% end
for i = 1 : txtNum
    sampleEntity = docEntity(i, :);
    capSum = sum(bsxfun(@times, sampleEntity, docEntity), 2);
    temp = sum(docEntity, 2) + sum(sampleEntity) - capSum;
    W(i, :) = capSum ./temp;
end

W(isnan(W)) = 0;
W = W - diag(diag(W)) + eye(size(W));
txtSimMatrix = txtSimMatrix .* W;

txtSimMatrix = txtSimMatrix ./ repmat(sum(txtSimMatrix), txtNum, 1);
txtSimMatrix0 = -1 * txtSimMatrix0 ./ repmat(sum(txtSimMatrix0), txtNum, 1);
imgSimMatrix = imgSimMatrix ./ repmat(sum(imgSimMatrix), txtNum, 1);
imgSimMatrix0 = -1 * imgSimMatrix0 ./ repmat(sum(imgSimMatrix0), txtNum, 1);
txtSimMatrix = txtSimMatrix + txtSimMatrix0;
imgSimMatrix = imgSimMatrix + imgSimMatrix0;

txtSimMatrix = (txtSimMatrix + txtSimMatrix') / 2;

imgSimMatrix = txtSimMatrix;
simAll = [simMatrix, txtSimMatrix; imgSimMatrix, simMatrix];
% do norm
% for i = 1 : size(simAll, 1)
%     posNum = sum(simAll(i, :) == 1);
%     negNum = sum(simAll(i, :) == -1);
%     
%     div = ones(size(simAll(i, :)));
%     div(simAll(i, :) == -1) =  posNum / negNum;
%     simAll(i, :) = simAll(i, :) .* div;
% end

DAll = diag(sum(simAll, 2));
LAll = DAll - simAll;
L11 = LAll(1 : txtNum, 1 : txtNum);
L12 = LAll(txtNum+1 : end, 1 : txtNum);
L21 = LAll(1:txtNum, txtNum+1:end);
L22 = L11;

alpha = 1;
beta = 1;
mu = 1;
for i = 1 : iterMaxNum
%     objValue = alpha * (trace((T_tr*P_T)' * (DMatrix - simMatrix) * I_tr * P_I)) + ...
%         sum(sum((T_tr * P_T - I_tr * P_I) .^ 2)) + ...
%         mu * (sum(sum((T_tr * P_T - Y).^2)) + sum(sum((I_tr * P_I - Y).^2))) + ...
%         lambda * ( sum(sum(P_T.^2)) + sum(sum(P_I.^2)) );
%     objValue = alpha * trace((T_tr*P_T)' * L11 * (T_tr*P_T) + (I_tr*P_I)' * L22 * (I_tr*P_I) + ...
%        (T_tr*P_T)'*L12*(I_tr*P_I) + (I_tr*P_I)'*L21*(T_tr*P_T)) + ...
%        sum(sum((T_tr * P_T - I_tr * P_I) .^ 2)) + ...
%         mu * (sum(sum((T_tr * P_T - Y).^2)) + sum(sum((I_tr * P_I - Y).^2))) + ...
%         lambda * ( sum(sum(P_T.^2)) + sum(sum(P_I.^2)) );
    objValue = alpha * trace((T_tr*P_T)' * L11 * (T_tr*P_T) + (I_tr*P_I)' * L22 * (I_tr*P_I) + ...
       (T_tr*P_T)'*L12*(I_tr*P_I) + (I_tr*P_I)'*L21*(T_tr*P_T)) + ...     
        mu * (sum(sum((T_tr * P_T - Y).^2)) + sum(sum((I_tr * P_I - Y).^2))) + ...
        lambda1 * sum(sum(P_T.^2)) + lambda2 * sum(sum(P_I.^2));
    
%     term1 = alpha * trace((T_tr*P_T)' * L11 * (T_tr*P_T) + (I_tr*P_I)' * L22 * (I_tr*P_I) + ...
%        (T_tr*P_T)'*L12*(I_tr*P_I) + (I_tr*P_I)'*L21*(T_tr*P_T));
%     term2 = sum(sum((T_tr * P_T - I_tr * P_I) .^ 2));
%     term3 = mu * (sum(sum((T_tr * P_T - Y).^2)) + sum(sum((I_tr * P_I - Y).^2)));
%     term4 = lambda1 * sum(sum(P_T.^2)) + lambda2 * sum(sum(P_I.^2));
%     
%     term1_ori = trace((T_tr*P_T)' * L11 * (T_tr*P_T) + (I_tr*P_I)' * L22 * (I_tr*P_I) + ...
%        (T_tr*P_T)'*L12*(I_tr*P_I) + (I_tr*P_I)'*L21*(T_tr*P_T));
%     term2_ori = sum(sum((T_tr * P_T - I_tr * P_I) .^ 2));
%     term3_ori = sum(sum((T_tr * P_T - Y).^2)) + sum(sum((I_tr * P_I - Y).^2));
%     term4_ori = sum(sum(P_T.^2)) + sum(sum(P_I.^2));

    ratio = abs((lastObjValue - objValue) / lastObjValue);
%     fprintf('Iter %d: term1=%.2f term2=%.2f term3=%.2f term4=%.2f\n', i, term1_ori, term2_ori, term3_ori, term4_ori);
%     fprintf('Iter %d: term1=%.2f term2=%.2f term3=%.2f term4=%.2f\t', i, term1, term2, term3, term4);
    fprintf('Iter %d: lossValue = %f\t ratio=%f\n', i, objValue, ratio);
    
    if lastObjValue ~= 0 && (ratio < 0.001 || lastObjValue < objValue)
        break;
    end
    lastObjValue = objValue;

%     P_T_next = inv(lambda1 * eye(txtDim, txtDim) + mu * (T_tr') * T_tr + alpha * T_tr'*L11*T_tr) * ...
%         (mu *T_tr' * Y - 0.5 * alpha * (T_tr'*L21'*I_tr*P_I + T_tr'*L12*I_tr*P_I));
    P_T_next = (lambda1 * eye(txtDim, txtDim) + mu * (T_tr') * T_tr + alpha * T_tr'*L11*T_tr) \ ...
        (mu *T_tr' * Y - 0.5 * alpha * (T_tr'*L21'*I_tr*P_I + T_tr'*L12*I_tr*P_I));

%     P_I_next = inv(lambda2 * eye(imgDim, imgDim) + mu * (I_tr') * I_tr + alpha * I_tr'*L22*I_tr) * ...
%         (mu *I_tr' * Y - 0.5 * alpha * (I_tr'*L12'*T_tr*P_T + I_tr'*L21*T_tr*P_T));
    P_I_next = (lambda2 * eye(imgDim, imgDim) + mu * (I_tr') * I_tr + alpha * I_tr'*L22*I_tr) \ ...
        (mu *I_tr' * Y - 0.5 * alpha * (I_tr'*L12'*T_tr*P_T + I_tr'*L21*T_tr*P_T));
    
    P_T = P_T_next;
    P_I = P_I_next;
end