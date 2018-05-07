% helinger norm
function [tr_norm, te_norm] = hnorm(tr,te)

div_tr = repmat(sum(tr, 2), 1, size(tr, 2));
tr = tr ./ (div_tr + eps);
tr = tr.^0.5;

div_te = repmat(sum(te, 2), 1, size(te, 2));
te = te ./ (div_te + eps);
te = te.^0.5;

tr_n = size(tr,1);
te_n = size(te,1);
tr_mean = mean(tr,1);
tr_std = std(tr,1);
% tr_mean = mean([tr;te],1);
% tr_std = std([tr;te],1);
tr_norm = (tr-repmat(tr_mean,tr_n,1))./(repmat(tr_std,tr_n,1) + eps);
te_norm = (te-repmat(tr_mean,te_n,1))./(repmat(tr_std,te_n,1) + eps);