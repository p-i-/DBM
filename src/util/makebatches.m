function [batchdata_train, batchtargets_train,   ...
          batchdata_test , batchtargets_test ] ...
= makebatches( data_folder, nBatchSize )

    [batchdata_train, batchtargets_train] = makebatches_( data_folder, 'train', nBatchSize );
    [batchdata_test , batchtargets_test ] = makebatches_( data_folder, 'test' , nBatchSize );
end

function [batchdata, batchtargets] ...
= makebatches_( data_folder, strDataSet, nBatchSize )
    images = [];
    targets = [];
    imageFile = @(k) [data_folder strDataSet num2str(k) '.mat'];
    for k = 0 : 9
        load( imageFile(k) ); % loads 'D'
        
        targ = circshift( [1 0 0 0 0 0 0 0 0 0], k, 2 );  %disp(targ);
        targs = repmat( targ, size(D,1), 1 );
        
        images  = [images ; D    ];
        targets = [targets; targs];
    end
    
    totItems = size(images,1);
    N = nBatchSize;
    nBatches = floor( totItems / N );
    nPixels = size( images, 2 );

    fprintf(1, 'Shuffling %s dataset (%d items) into %d batches of %d.\n', ...
        strDataSet, totItems, nBatches, N);

    batchdata    = zeros(N, nPixels, nBatches);
    batchtargets = zeros(N,      10, nBatches);

    rand('state', 0); %so we know the permutation of the training data
    randomorder = randperm(totItems);
    
    for b = 1 : nBatches
      batchdata   (:,:,b) =  images( randomorder(1+(b-1)*N:b*N ), : );
      batchtargets(:,:,b) = targets( randomorder(1+(b-1)*N:b*N ), : );
    end;
    
    batchdata = batchdata / 255.0; % get in range 0.0 to 1.0
    
    % Reset random seeds 
    rand ('state',sum(100*clock)); 
    randn('state',sum(100*clock)); 
end
