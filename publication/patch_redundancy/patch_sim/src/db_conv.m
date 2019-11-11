function [crossed_term] = db_conv(h,auto_mask,auto_h_L)

L_h = size(h,1);
L = size(auto_mask,1) - L_h;
crossed_term = zeros([L+L_h L+L_h size(h,3)]);

for lin_t = 1:L_h^2
    [tx,ty] = ind2sub([L_h L_h], lin_t);
    ttx = tx - 1;
    tty = ty - 1;
    auto_h_L1 = circshift(circshift(auto_h_L,ttx,1),tty,2);
    auto_h_L2 = circshift(circshift(auto_h_L,-ttx,1),-tty,2);
    prod = auto_mask .* auto_h_L1 .* auto_h_L2;
    crossed_term(tx,ty,:) = sum(sum(prod,1),2);
end

