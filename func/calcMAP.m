%% calculate cross-modal retrieval MAP
function [imgQueryMAP, txtQueryMAP, imgQueryCatMAP, txtQueryCatMAP] = calcMAP(T_tr, T_te, I_tr, I_te, trTxtCat, trImgCat, teImgCat, teTxtCat, catNum, k, distType)

% txtNum X imgNum
simMatrix = calcSimByDist(T_te, I_te, distType);

% fusion
% simMatrix1 = calcSimByDist(T_te, I_te, 4);
% simMatrix2 = calcSimByDist(T_te, I_te, 5);
% simMatrix = (simMatrix1 + simMatrix2) / 2;

% simMatrix = calcSimByKNN(T_tr, T_te, I_tr, I_te, trTxtCat, trImgCat, catNum, k, distType); 
% T_tr_num = size(T_tr, 1);
% T_te_num = size(T_te, 1);
% I_tr_num = size(I_tr, 1);
% I_te_num = size(I_te, 1);
% totalNum = size(T_tr, 1) + size(T_te, 1) + size(I_tr, 1) + size(I_te, 1);
% totalCat = [trTxtCat; teTxtCat; trImgCat; teTxtCat];
% Y_init = zeros(totalNum, 10);
% for i =  1 : T_tr_num + I_tr_num
%     Y_init(i, totalCat(i)) = 1;
% end
% simMatrix = calcSimByTagProp(T_tr, T_te, I_tr, I_te, Y_init, 0.5, 0.01);

% simMatrix = calcSimBySVM(T_tr, T_te, I_tr, I_te, trTxtCat, trImgCat, catNum, distType);

%% calculate category
txtNum = size(simMatrix, 1);
imgNum = size(simMatrix, 2);

imgQueryCat = zeros(imgNum, txtNum);
txtQueryCat = zeros(txtNum, imgNum);
 
[~, indImgQuery] = sort(simMatrix, 'descend');
[~, indTxtQuery] = sort(simMatrix, 2, 'descend');

for i = 1 : imgNum
    imgQueryCat(i, :) = teTxtCat(indImgQuery(:, i));
end
for i = 1 : txtNum
    txtQueryCat(i, :) = teImgCat(indTxtQuery(i, :));
end

topRank = 0;
if topRank == 0
    topRank = txtNum;
end

txtCatNums =  zeros(catNum, 1);
imgCatNums =  zeros(catNum, 1);
for i = 1 : catNum
    txtCatNums(i) = sum(teTxtCat == i);
    imgCatNums(i) = sum(teImgCat == i);
end

%% calculate MAP
% % more time-consuming
% imgQueryAPs = zeros(imgNum, 1);
% for i = 1 : imgNum
%     idx = find(imgQueryCat(i, :) == teImgCat(i));
% 	count = length(idx);
% 	imgQueryAPs(i) = mean((1 : count) ./ idx);
% end
% txtQueryAPs = zeros(imgNum, 1);
% for i = 1 : imgNum
%     idx = find(txtQueryCat(i, :) == teTxtCat(i));
% 	count = length(idx);
% 	txtQueryAPs(i) = mean((1 : count) ./ idx);
% end

imgQueryAPs = zeros(imgNum, 1);
for i = 1 : imgNum
    temp = imgQueryCat(i, 1:topRank) == teImgCat(i);
    count = 0;
    for j = 1 : length(temp)        
        if temp(j)
            count = count + 1;            
            imgQueryAPs(i) = imgQueryAPs(i) + count / j;
        end
    end
    
    if count == 0
        imgQueryAPs(i) = 0;
    else
        imgQueryAPs(i) = imgQueryAPs(i) / count;
    end
end

txtQueryAPs = zeros(txtNum, 1);
for i = 1 : txtNum
    temp = txtQueryCat(i, 1:topRank) == teTxtCat(i);    
    count = 0;
    for j = 1 : length(temp)
        if temp(j)
            count = count + 1;            
            txtQueryAPs(i) = txtQueryAPs(i) + count / j;
        end
    end
    
    if count == 0
        txtQueryAPs(i) = 0;
    else
        txtQueryAPs(i) = txtQueryAPs(i) / count;
    end
end


imgQueryCatMAP = zeros(catNum, 1);
txtQueryCatMAP = zeros(catNum, 1);
for i = 1 : catNum
	temp = imgQueryAPs(teImgCat == i);
	imgQueryCatMAP(i) = sum(temp) / imgCatNums(i);
	
	temp = txtQueryAPs(teTxtCat == i);
	txtQueryCatMAP(i) = sum(temp) / txtCatNums(i);
end

imgQueryMAP = mean(imgQueryAPs);
txtQueryMAP = mean(txtQueryAPs);