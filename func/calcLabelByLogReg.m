function [Y_txt_te, Y_img_te, Y_txt_tr, Y_img_tr, predTrCat, predTeCat] = calcLabelByLogReg(T_tr, T_te, I_tr, I_te, trTxtCat, trImgCat, teTxtCat, teImgCat, catNum)

T_tr_num = size(T_tr, 1);
T_te_num = size(T_te, 1);
I_tr_num = size(I_tr, 1);
I_te_num = size(I_te, 1);
totalNum = T_tr_num + T_te_num + I_tr_num + I_te_num;
%% train

% generate train label
train_label = [trTxtCat; trImgCat];

% generate train feature
train_fea = sparse([T_tr; I_tr]);

liblinear_train_param = '-s 0 -c 1000 -q'; % -c 2000
% liblinear_train_param = '-s 0 -c 1 -B 1 -q'; % for SCP
if exist('model.mat', 'file') == 0
    model = train(train_label, train_fea, liblinear_train_param);
%     save('model', 'model');
else
    load('model');
end

%% predict
test_label = [teTxtCat; teImgCat];
test_fea = sparse([T_te; I_te]);
liblinear_predict_param = '-b 1';
fprintf('Test ');
[predTeCat, accuracy, prob_estimates_te] = predict(test_label, test_fea, model, liblinear_predict_param);
fprintf('Train ');
[predTrCat, accuracy, prob_estimates_tr] = predict(train_label, train_fea, model, liblinear_predict_param);
%% generate test prob
prob_te = zeros(T_te_num+I_te_num, catNum);
for i = 1 : catNum
    prob_te(:, i) = prob_estimates_te(:, model.Label == i);
end

Y_txt_te = prob_te(1:T_te_num, :);
Y_img_te = prob_te(T_te_num+1 : end, :);

% confMatrix = constructConfMatrix( predTeCat, test_label, catNum );
% save('./result/Wiki/confMatrix_DPEP', 'confMatrix');

%% generate train prob
prob_tr = zeros(T_tr_num+I_tr_num, catNum);
for i = 1 : catNum
    prob_tr(:, i) = prob_estimates_tr(:, model.Label == i);
end

Y_txt_tr = prob_tr(1:T_tr_num, :);
Y_img_tr = prob_tr(T_tr_num+1 : end, :);