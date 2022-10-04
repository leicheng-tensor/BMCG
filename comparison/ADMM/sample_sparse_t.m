%SAMPLE_SPARSE_T:   transpose of sparse sampling operator A. Returns a
%                   matrix that in the positions specified by mask contains
%                   the rows of the input matrix, and zeros elsewhere.
%
%X = sample_sparse_t(X_in, mask)
%
%
%   X = A^T * X_in, where A = sample_sparse(I,mask)
%
%example:
% sample_sparse_t([1 2 3; 4 5 6], logical([0, 1, 0, 1]))
% 
% ans =
% 
%      0     0     0
%      1     2     3
%      0     0     0
%      4     5     6
%      
%
%
%see also: sample_sparse, sample_sparse_AtA
%
%
%code author: Vassilis Kalofolias
%date: July 2013


function X = sample_sparse_t(X_in, mask)

[m, n] = size(X_in);

if not(islogical(mask))
    error('mask has to be logical')
elseif nnz(mask) ~= m
    error('non zero elements of mask have to be equal to number of rows of A_in')
end

if isa(X_in, 'single')
    X = zeros(length(mask), n, 'single');
else
    X = zeros(length(mask), n);
end
X(mask, :) = X_in;

