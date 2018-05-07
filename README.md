## Introduction
This is the source code of our Neurocomputing 2017 paper "[Cross-media Retrieval by Exploiting Fine-Grained Correlation at Entity Level](http://59.108.48.34/tiki/download_paper.php?fileId=344)", please cite the following paper if you use our code.

    Lei Huang and Yuxin Peng, "Cross-media Retrieval by Exploiting Fine-Grained Correlation at Entity Level", Neurocomputing, Vol. 236, pp. 123-133, May. 2017.


## Usage

Dataset——Wikipedia

Directory：./data/Wiki

Text feature：./data/Wiki/LDA_200_Gibbs2000.mat （200 dimension）

Image feature：./data/Wiki/I_BOW_4096_vlfeat.mat （4096 dimension）

Training label：./data/Wiki/trainset_txt_img_cat.list

Testing label：./data/Wiki/testset_txt_img_cat.list

Training set：2173 image/text pairs
Testing set：693 image/text pairs


Run code：./DPEP/mainWiki.m

## Related Link
Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
