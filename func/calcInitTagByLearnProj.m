function initTag = calcInitTagByLearnProj(P_T, P_I, testFea, mediaType, k)
 
% testFea: row vector
if mediaType == 1 % text
    initTag = testFea * P_T;
    dim = size(P_T, 2);
else % image
    initTag = testFea * P_I;
    dim = size(P_I, 2);
end

% knn process to optimize
teNum = size(testFea, 1);
if exist('k') == 0
    k = dim;
end
if k > dim
    k = dim;
end

initTag_knn = zeros(size(initTag));
for i = 1 : teNum
    [K_txt, ind_txt] = sort(initTag(i, :), 'descend');        
    ind = ind_txt(1 : k);
    initTag_knn(i, ind) = initTag(i, ind); 
end
initTag = initTag_knn;