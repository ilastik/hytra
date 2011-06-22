%%% all zeros uin16 volume
matfilename = 'all_zeros_stack.mat'
maxdims = 17

dims = unidrnd(maxdims, [1,3])
stack = uint16(zeros(dims))
save(matfilename, 'stack')
