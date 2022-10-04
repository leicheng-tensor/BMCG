% SPLIT_OBSERVED: Keep observed values and separate into different vectors.
% Useful for tasks where we have missing values and we want to split the
% observed ones into train / validation / test parts
%
% [y1, m1, y2, m2, ...] = split_observed(Y, ratio, ignore_val, rand_seed)
%
% INPUTS: 
%       
%       Y: the vector / matrix / array to be split
%       ratio: either a number from (0,1] or a vector giving the proportion
%               of the number of elements in each of the returned parts
%       ignore_val: the value that indicates a missing value. Default is
%               NaN, while for sparse matrices 0 should be used. The  
%               elements of Y with this value are not considered in the
%               splitting.
%       rand_seed: The seed used for initializing the random number
%               generator. Useful for being able to reproduce the same mask
%               over different calls for the same inputs (at least on the
%               same system). Default: 0.
%
%
% OUTPUTS: 
%       y1: vector containing part 1
%       m1: mask vector that indicates which elements of input Y reside in
%           y1. This means that Y(m1) == y1
%       y2, m2: same as above.
%
%
%  NOTE THAT if Y is sparse, m1, m2,... will also be sparse, but y1, y2,
%       ... will always be full vectors!
%
% Example: 
%   
% Y = randi(5,3,5)
% Y(3) = nan
% % keep all the observed values in vector y
% y = split_observed(Y, 1)
% % split in two the values, considering NaN as unobserved value
% [y1, m1, y2, m2] = split_observed(Y, .8);
% [m1, m2, isnan(Y(:))]
% % split in two, considering 2 as an unobserved value
% [y1, m1, y2, m2] = split_observed(Y, .8, 2);
% [m1, m2, Y(:)==2]
%
%
%
% see also: split_train_CV, sample_sparse, sample_sparse_t,
%           sample_sparse_AtA 
%
% code author: Vassilis Kalofolias
% date: Feb 2014




function [varargout] = split_observed(Y, ratio, ignore_val, rand_seed)

if nargin < 3
    ignore_val = nan;
end

if nargin < 4
    rand_seed = [];
end

if length(ratio) == 1
    if ratio > 1 || ratio < 0
        error('ratio has to be between 0 and 1 or a vector of positive numbers');
    else
        ratio = [ratio, 1-ratio];
    end
end

if not(isempty(rand_seed))
    set_seed(rand_seed);
end

n = numel(Y);

if isnan(ignore_val)
    [ind_all2obs] = find(not(isnan(Y(:))));
else
    [ind_all2obs] = find(not(Y(:) == ignore_val));
end

n_obs = length(ind_all2obs);

% index that takes observations and shuffles them
ind_obs2shuffled = randperm(n_obs);

% cut the observed values in the intervals given by the ratio
lims = [0 cumsum(ratio)];
lims = round(n_obs * lims / lims(end));     % map this to [0 ... n_obs]

varargout = cell(nargout, 1);

for i = 1:(min(length(ratio), ceil(nargout/2)))
    
    mask = false(n, 1);
    mask(ind_all2obs(ind_obs2shuffled( (1 + lims(i)) : lims(i+1)))) = true;
    
    varargout{2*i - 1} = full(Y(mask)); 
    varargout{2*i} = mask;
end







