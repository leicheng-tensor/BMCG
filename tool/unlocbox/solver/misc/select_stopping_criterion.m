function stop_crit = select_stopping_criterion(algo)
%
%
%   This function select a default stopping criterion
%
%   Url: https://epfl-lts2.github.io/unlocbox-html/doc/solver/misc/select_stopping_criterion.html

% Copyright (C) 2012-2016 Nathanael Perraudin.
% This file is part of UNLOCBOX version 1.7.5
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

    if isstr(algo)
        switch lower(algo)
            case 'admm'
                stop_crit = 'rel_norm_primal_dual';
            case 'chambolle_pock'
                stop_crit = 'rel_norm_primal_dual';
            case 'sdmm'
                stop_crit = 'rel_norm_primal_dual';
            case 'pocs'
                stop_crit = 'obj_threshold';
            case 'fb_based_primal_dual'
                stop_crit = 'rel_norm_primal_dual';  
            case 'fbf_primal_dual'
                stop_crit = 'rel_norm_primal_dual';  
            otherwise
                stop_crit = 'rel_norm_obj';
        end
    else
        stop_crit = 'rel_norm_obj';
    end
end
