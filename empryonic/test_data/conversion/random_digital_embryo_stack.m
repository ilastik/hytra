%%% random uint16 volume
matfilename = 'random_stack.mat'
maxdims = 17

dims = unidrnd(maxdims, [1,3])
stack = uint16(unidrnd(2^16,dims)-1)
save(matfilename, 'stack')

