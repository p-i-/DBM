function sample = binarySampleFromPDs( pd )
    [batchSize_, nOutputs] = size( pd );

    bools = cumsum(pd,2) > repmat( rand(batchSize_,1), 1, nOutputs );

    % e.g. 001 111 gives (6+1) - 4 = 3
    indexOfFirstONE = (nOutputs+1) - sum(bools, 2);

    sample = 0 .* pd;  
    sample( ...
        sub2ind( size(pd), 1:batchSize_, indexOfFirstONE' ) ...
        ) = 1;    
end
