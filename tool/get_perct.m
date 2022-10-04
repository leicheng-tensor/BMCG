%Calculate the same percentage
%Input	A: original matrix
%       As:matrix predicted
%       idx_pair: the pairs to compute

function  same_perct = get_perct(A, As, idx_pair)
if nargin<3
    idx_pair = 1:1:(size(A,1)*size(A,2));
end

temp_A = A(idx_pair);
temp_As= As(idx_pair);
temp_As(temp_As< min(min(A)))=min(min(A));
temp_As(temp_As>=max(max(A)))=max(max(A));
temp_As=round(temp_As);

same_perct = length(find(temp_As==temp_A >0))/length(idx_pair);

end