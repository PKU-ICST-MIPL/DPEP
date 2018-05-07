# DPEP
Source code of our Neurocomputing 2017 paper "Cross-media Retrieval by Exploiting Fine-Grained Correlation at Entity Level".

Dataset——Wikipedia

Directory：./data/Wiki
Text feature：./data/Wiki/LDA_200_Gibbs2000.mat （200 dimension）
Image feature：./data/Wiki/I_BOW_4096_vlfeat.mat （4096 dimension）
Training label：./data/Wiki/trainset_txt_img_cat.list
Testing label：./data/Wiki/testset_txt_img_cat.list

Training set：2173 image/text pairs
Testing set：693 image/text pairs


Run code：./DPEP/mainWiki.m
