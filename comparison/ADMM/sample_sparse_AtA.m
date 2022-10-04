%SAMPLE_SPARSE_ATA:   gram matrix A^t*A of sparse sampling operator A.
%                   Zeros-out the rows of the matrix corresponding to zeros
%                   in the mask given.
%
% X = sample_sparse_AtA(X_in, mask)
%
% equivalent to (but cheaper than)
% X = sample_sparse_t(sample_sparse(X_in, mask), mask)
%
%   X = At * A * X_in, where A = sample_sparse(I, mask)
%                            At = sample_sparse_t(I, mask)
%
%example:
% sample_sparse_AtA([1 2 3; 4 5 6; 7 8 9; 1 3 5], logical([0, 1, 0, 1]))
% 
% ans =
% 
%      0     0     0
%      4     5     6
%      0     0     0
%      1     3     5
%      
%
%
%see also: sample_sparse, sample_sparse_t
%
%
%code author: Vassilis Kalofolias
%date: October 2013


function X_in = sample_sparse_AtA(X_in, mask)

if not(islogical(mask))
    error('mask has to be logical')
elseif length(mask) ~= size(X_in, 1);
    error('elements of mask have to be equal to number of rows of A_in')
end

X_in(~mask, :) = 0;

