function poly = cubeToPoly(c)
NUM_STATE_VARS = size(c,1);
Al = -eye(NUM_STATE_VARS);
bl = -c(:,1);
Ah = eye(NUM_STATE_VARS);
bh = c(:,2);

% collect indices of elements in b with infinity as the value
arrInf = [];
% remove all constraints such as x > -inf
for i = 1:NUM_STATE_VARS
    if bl(i) == inf %|| poly.b(i) == -inf, redundant. will never occur
        arrInf = [arrInf i];
    end
end

Al(arrInf,:) = 0;
bl(arrInf) = 0;

arrInf = [];
% remove all constraints such as x < inf
for i = 1:NUM_STATE_VARS
    if bh(i) == inf %|| poly.b(i) == -inf, redundant. will never occur
        arrInf = [arrInf i];
    end
end
Ah(arrInf,:) = 0;
bh(arrInf) = 0;

poly.A = [Al;Ah];
poly.b = [bl;bh];
end
