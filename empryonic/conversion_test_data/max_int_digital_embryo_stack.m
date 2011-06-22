%%% max int uin16 volume
matfilename = 'max_int_stack.mat'
maxdims = 17

dims = unidrnd(maxdims, [1,3])
stack = uint16(ones(dims)*(2^16-1))
save(matfilename, 'stack')
