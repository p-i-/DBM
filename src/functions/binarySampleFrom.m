function sample = binarySampleFrom( pd ) % prob dist
    sample = pd > rand( size(pd) );
end
