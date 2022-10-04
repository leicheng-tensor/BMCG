% AVERAGE_ERROR compute average absolute and squared error (MAE, RMSE)
%
% It first maps inputs linearly from lims_in to [1,5] and then returns the
% per-element average absolute error and squared error
%
% Used for matrix completion for ratinge in the range of 1-5 stars. 
%
%USAGE:
%
%[MAE, RMSE] = average_error(y_pred, y_true, lims_init, lims_scaled, round_pred)
%
%
%INPUTS: 
%       y_pred: predicted ratings
%       y_true: ground truth ratings in [lims_init]
%       lims_init: the ratings valid range before any scaling
%       lims_scaled: the ratings range after scaling: we first fit the data
%                   from the range lims_init to the range lims_scaled and
%                   then compute the error.
%       round:  {0, 1}. If true, round the real-valued ratings to the
%               nearest integer value after first truncating to the valid
%               range before rescaling.
%
%
%outputs:
%
% MAE: mean absolute error              mean(abs(x-y))
% RMSE: root mean squared error         sqrt(mean(x-y)^2)
%
%
%
% see also: lin_map, MC_demo_solve_ADMM, MC_solve_ADMM
%
% code author: Vassilis Kalofolias
% date: Feb 2014

function [MAE, RMSE] = average_error(y_pred, y_true, lims_init, lims_scaled, round_pred)




y_pred = lin_map(y_pred, lims_init, lims_scaled);
y_true = lin_map(y_true, lims_init, lims_scaled);

if round_pred
    y_pred(y_pred < lims_init(1)) = lims_init(1);
    y_pred(y_pred > lims_init(2)) = lims_init(2);
    y_pred = round(y_pred);
end
    


MAE = mean(abs(y_pred - y_true));
RMSE = sqrt(mean((y_pred - y_true).^2)); % = ||y_pred - y_true|| / sqrt(N)









