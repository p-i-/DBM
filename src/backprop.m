function backprop( maxepoch, ...
        batchdata_train, batchtargets_train, ...
        batchdata_test, batchtargets_test, ...
        W01, W12, W32, b1, b2, b3 )
        
    fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
    fprintf(1,'60 batches of 1000 cases each. \n');

    [~, neurons0, nBatches_train] = size( batchdata_train ); % batchSize
    
    [~, neurons2] = size(W12); % neurons1
        
    fprintf( 1, 'TRAINING\n' ); [pd2_train] = calcErrors( batchdata_train, batchtargets_train );
    fprintf( 1, 'TEST\n' );     [~        ] = calcErrors( batchdata_test , batchtargets_test );
    
    function [ pd2_all ] = ...
    calcErrors( batchData, batchTargets )
    
        err_ = zeros(maxepoch);
        crerr_ = zeros(maxepoch);
        
        [batchSize, ~, nBatches] = size( batchData );
        pd2_all = zeros( batchSize, neurons2, nBatches );
        
        for kBatch = 1 : nBatches
            pd0 = batchData( :, :, kBatch );
                        
            B1 = repmat(b1,batchSize,1);
            B2 = repmat(b2,batchSize,1);
            
            pd1 = sigmoid( pd0*2*W01 + B1 );
            pd2 = sigmoid( pd1*  W12 + B2 );
            
            for meanField = 1 : 50
                pd1 = sigmoid( pd0*W01 + B1 + pd2*W12' );
                pd2 = sigmoid( pd1*W12 + B2 );
            end
             
            pd2_all( :, :, kBatch ) = pd2;
        end

        for epoch_ = 1 : maxepoch

            B1 = repmat( b1, batchSize, 1 );
            B2 = repmat( b2, batchSize, 1 );
            B3 = repmat( b3, batchSize, 1 );

            err_cr = 0;
            nWrong = 0;
            for kBatch = 1 : nBatches
                pd0     = batchData(    :, :, kBatch );
                pd2_ALL = pd2_all(      :, :, kBatch );
                targets = batchTargets( :, :, kBatch );

                pd1 = sigmoid( pd0*W01 + B1 + pd2_ALL*W12' );
                pd2 = sigmoid( pd1*W12 + B2 );
                
                expZ = exp( pd2*W32' + B3 );
                pd3 = expZ ./ repmat( sum(expZ,2), 1, 10 );

                [~, index_pd  ] = max( pd3    , [], 2 );
                [~, index_targ] = max( targets, [], 2 );

                wrongs = nnz( index_pd ~= index_targ );
                cr = - sum(sum( targets .* log(pd3) ));
                
                
                % s1 = binarySampleFrom( pd0*W01 + B1 );
                % s2 = binarySampleFrom( s1*W12 + B2 );
                % expZ = exp( s2*W32' + B3 );
                % pd3 = expZ ./ repmat( sum(expZ,2), 1, 10 );
                % s3 = binarySampleFromPDs( pd3 );
                % 
                % correct = nnz( s3 .* targets );
                
                
                nWrong = nWrong + wrongs; %(batchSize-correct);
                err_cr = err_cr + cr;
            end
        
            err_(epoch_) = nWrong;
            crerr_(epoch_) = err_cr;
            fprintf( 1, ' Epoch %d misclass^n err: %d of %2dk,  X-entropy err %f \n', ...
                epoch_, nWrong, batchSize*nBatches/1000, err_cr );
        end
    end
    
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    W21 = W12';
    W23 = W32';
    
    % work in blocks of 100 batches
    batchesInBlock = 100;
    
    nBlocks = nBatches_train / batchesInBlock;
    
    fprintf( 1, '\n\nPerforming Conjugate Gradient Optimization:\n' );
    for epoch = 1 : maxepoch
        fprintf( 1, '\n    Epoch %d, batch-block: ', epoch );
        
        shuffledBatches = randperm( nBatches_train );
        intoBlocks = reshape( shuffledBatches, nBlocks, [] );
                
        for b = 1 : nBlocks
            %fprintf(1, ' %d', b);            
            
            block = intoBlocks( b, : ); % 1 x batchesInBlock: each elt represents a batch#
            
            data    = reshape(   batchdata_train(:,:,block), [], size(   batchdata_train,2) );
            pd2_all = reshape(         pd2_train(:,:,block), [], size(         pd2_train,2) );
            targets = reshape(batchtargets_train(:,:,block), [], size(batchtargets_train,2) );
            
            %%%%%%%% DO CG with 3 linesearches
            
            % pack network params into a [D, 1] column-vector 
            %   (& remember dims of orig objects so we can unpack later)
            [theta, sizes] = flatten( { W01, W12, W23, W21, b1, b2, b3 } );
            
            do_init = epoch < 3;
            max_steps = 3;
            
            [X, ~] ...
                = minimize(  max_steps, @conjGrad, theta,  sizes, data, targets, pd2_all, do_init );
            
            N = norm( X - theta);
            fprintf( 'Norm: %f\n', N );
            
            % checkgrad(conjGrad,X,10^-5,sizes,data,targets);
            
            cell = unflatten( X, sizes );
            [ W01, W12, W23, W21, b1, b2, b3 ] = cell{:};
        end
        
    end

    
    fprintf( '\nDone!\n' );
    
    fprintf( 1, 'TRAINING\n' ); [~] = calcErrors( batchdata_train, batchtargets_train );
    fprintf( 1, 'TEST\n' );     [~] = calcErrors( batchdata_test , batchtargets_test );
    
end


function [f, df] ...
  = conjGrad( theta,  sizes, data, target, pd2_temp, do_init )
    
    cell = unflatten( theta, sizes );
    [ W01, W12, W23, W21, b1, b2, b3 ] = cell{:};

    batchSize = size(data,1);

    B1 = repmat(b1,batchSize,1);
    B2 = repmat(b2,batchSize,1);
    B3 = repmat(b3,batchSize,1);

    pd1 = sigmoid( data*W01 + B1 + pd2_temp*W21  ); % DBM09 3.3 explains this (I think?!)
    pd2 = sigmoid(  pd1*W12 + B2 );
    expZ = exp(     pd2*W23 + B3 );
    pd3 = expZ ./ repmat( sum(expZ,2), 1, 10 );

    f = - sum(sum( target .* log(pd3) ));

    IO = pd3 - target;

    Ix3 = IO;                            dw23 =      pd2'*Ix3;  db3 = sum(Ix3);
    Ix2 = (Ix3*W23') .* pd2 .* (1-pd2);  dw12 =      pd1'*Ix2;  db2 = sum(Ix2);
    Ix1 = (Ix2*W12') .* pd1 .* (1-pd1);  dw01 =     data'*Ix1;  db1 = sum(Ix1);
                                         dw21 = pd2_temp'*Ix1;  

    if do_init
        dw01 = 0 * dw01;
        dw12 = 0 * dw12;  dw21 = 0 * dw21;
        dw23 = 0 * dw23;

        db1 = 0 * db1;
        db2 = 0 * db2;
        db3 = 0 * db3;
    end
    
    %df = [dw01(:)' dw21(:)' dw12(:)' dw23(:)' db1(:)' db2(:)' db3(:)']';
    [df, ~] = flatten( { dw01, dw12, dw23, dw21, db1, db2, db3 } );
end
