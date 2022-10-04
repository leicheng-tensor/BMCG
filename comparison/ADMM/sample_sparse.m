%SAMPLE_SPARSE: sparse sampling operator A. Returns a matrix that contains
%               the rows specified by the indices given.
%               X = A*X_in
%
%X = sample_sparse(X_in, mask)
%
%
% example:
% sample_sparse(eye(4), logical([0, 1, 0, 1]))
% 
% ans =
% 
%      0     1     0     0
%      0     0     0     1
%
%see also: sample_sparse_t, sample_sparse_AtA
%
%
%code author: Vassilis Kalofolias
%date: July 2013


function X = sample_sparse(X, mask)

if size(X,1) ~= length(mask)
    error('size of mask not compatible with number of rows of X');
elseif not(islogical(mask))
    error('mask has to be logical')
end

X = X(mask, :);

