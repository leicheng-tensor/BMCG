clear
clc
load MC_community_example Gu Gm Xn

Gc = Gu;        % columns graph
Gr = Gm;        % rows graph

%% best settings found for 20% observations (with grid search): 
%% [n=3,   no graphs]:          error = 0.97
%% [n=.01, r=.003, c=.003]:     error = 0.88

% Keep 20% for training, the rest for validation
[y_train, mask_train, y_val, mask_val, y_test, mask_test] = split_observed(Xn, [.2, .8, 0]);

params.size_X = size(Xn);
X = Xn;
[M,N] = size(X);
mask = (reshape(mask_train,100,200));