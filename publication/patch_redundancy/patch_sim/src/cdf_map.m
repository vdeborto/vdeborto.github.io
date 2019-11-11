function [cdf_st,thres_st,icdf_st] = cdf_map(sim_st, m, var, p, method, nfa)

valprob = nfa/(size(sim_st,1)-p)^2;
if ndims(sim_st) == ndims(m)+1
    m = repmat(m,1,1,size(sim_st,3));
    var = repmat(var,1,1,size(sim_st,3));
end

switch method
  case {'L2','L2asymp'}
    cdf_st = normcdf(sim_st,m,sqrt(var));
    icdf_st = icdf('norm', nfa, m, sqrt(var));
  case {'ps','cos'}
    cdf_st = normcdf(-sim_st,-m,sqrt(var));
    icdf_st = -icdf('norm', valprob, -m, sqrt(var));
end

thres_st = cdf_st < valprob;