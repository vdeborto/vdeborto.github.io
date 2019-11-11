function [cdf_st,thres_st,icdf_st] = cdf_map_sphere(sim_st, p, nfa)

valprob = nfa/(size(sim_st,1)-p)^2;
h = 1 - sim_st .* (sim_st >= 0);
cdf_st = 0.5 * betainc(2 * h - h.^2, 0.5 * (p^2 - 1), 0.5);
thres_st = cdf_st < valprob;
int = betaincinv(2 * valprob, .5 * (p^2 - 1), .5);
icdf_st = 1 - sqrt(1 - int);