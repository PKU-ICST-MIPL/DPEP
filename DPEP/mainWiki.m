clear

%% configuration
addpath('..\func');
addpath('..\3rd-party');

dataDir = '..\data\Wiki';
trainCatTxt = [dataDir, '\trainset_txt_img_cat.list'];
testCatTxt = [dataDir, '\testset_txt_img_cat.list'];

img_Dim = 4096;
txt_Dim = 200;
cat_Num = 10;

postfix = '9163_MI_MEAN_5000';
entityGraphFile = sprintf('%s\\entityGraph_%s', dataDir, postfix);
entityDocFile = sprintf('%s\\entityDoc_%s', dataDir, postfix);

iterMaxNum = 100;
k = 1400;
lambda1 = 3000;
lambda2 = 20000;

%% read data
load([dataDir, '\LDA_200_Gibbs2000']);
load([dataDir, '\I_BOW_4096_vlfeat']);
[trainTxt trainImg trCat] = textread(trainCatTxt, '%s %s %d');
[testTxt testImg teCat] = textread(testCatTxt, '%s %s %d');

% do norm
[I_tr, I_te] = hnorm(I_tr,I_te);
[T_tr, T_te] = hnorm(T_tr,T_te);

load(entityGraphFile); % load variable: entityGraph
load(entityDocFile); % load variable; entityDoc

%% entity projection learning
tic;
[P_T, P_I] = learnProj_sspm2ne(entityDoc', entityDoc', trCat, trCat, T_tr, I_tr, iterMaxNum, lambda1, lambda2);
toc;

Y_txt_init = calcInitTagByLearnProj(P_T, P_I, T_te, 1, k);
Y_img_init = calcInitTagByLearnProj(P_T, P_I, I_te, 2, k);

%% evaluate 
Y_entity = entityDoc';
[Y_txt_te_lr, Y_img_te_lr, Y_txt_tr_lr, Y_img_tr_lr] = calcLabelByLogReg(Y_entity, Y_txt_init, Y_entity, Y_img_init, trCat,trCat, teCat, teCat, cat_Num);
[imgQueryMAP_lr, txtQueryMAP_lr, imgQueryCatMAP_lr, txtQueryCatMAP_lr] = calcMAP(Y_txt_tr_lr, Y_txt_te_lr,  Y_img_tr_lr, Y_img_te_lr, trCat,trCat, teCat, teCat, 10, 100, 4);
disp(['Image Query Text: ' num2str(imgQueryMAP_lr)]);
disp(['Text Query Image: ' num2str(txtQueryMAP_lr)]);