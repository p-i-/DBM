function dbm_mf( maxepoch, W01, W12, W32, b0, b1, b2, b3, ...
        batchdata_train, batchtargets_train )
            
    [batchSize, neurons3, nBatches] = size( batchtargets_train ); % ~ = num pixels
    
    % Initializing symmetric weights and biases. 
    dW01 = 0*W01;
    dW12 = 0*W12;
    dW32 = 0*W32;
        
    db0 = 0*b0;
    db1 = 0*b1;
    db2 = 0*b2;
    db3 = 0*b3;
        
    for epoch = 1 : maxepoch

        fprintf( 1, 'epoch %d ', epoch );
        cum_err = 0;
        
        nCorrect = 0;
        batchesDone = 0;
        for randomBatch = randperm( nBatches )
            batchesDone = batchesDone + 1;
            
            if ~mod(batchesDone,20), fprintf(1,'.'); end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get Deltas
            data    =    batchdata_train( :, :, randomBatch );
            targets = batchtargets_train( :, :, randomBatch );

            [dSynapse01, dSynapse12, dSynapse32, dAct0, dAct1, dAct2, dAct3, avNeuronErr] ...
                = getDeltas( data, targets );
            
            cum_err = cum_err + avNeuronErr;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            % UPDATE WEIGHTS AND BIASES
            
            momInitial  = 0.5;
            momFinal    = 0.9;    
            mom = (epoch<=5)*momInitial + (epoch>5)*momFinal;
            
            % This code also adds sparcity penalty
            kSparsetarget1 = .2;
            kSparsetarget2 = .1;
                        
            kDamping = .9;
            if batchesDone == 1
                avAct1_damped = A1_pos; % sparsetarget1 * ones(1,neurons1);
                avAct2_damped = A2_pos; % sparsetarget2 * ones(1,neurons2);
            end
            avAct1_damped = kDamping*avAct1_damped + (1-kDamping)*A1_pos;
            avAct2_damped = kDamping*avAct2_damped + (1-kDamping)*A2_pos;
            
            kSparseCost = .001;
            sparsegrads1 = kSparseCost * ( repmat(avAct1_damped,batchSize,1) - kSparsetarget1 );
            sparsegrads2 = kSparseCost * ( repmat(avAct2_damped,batchSize,1) - kSparsetarget2 );
            % ^ or: kSparseCost * (avAct1_damped - kSparsetarget1) * ones( 1, batchSize )
            
            if batchesDone == 1
                eps_w      = 0.001;   % Learning rate for weights
                eps_b0     = 0.001;   % Learning rate for biases of visible units
                eps_b3     = 0.001;   % Learning rate for biases of hidden units
            end
            eps_w  = max( eps_w  * .999985, 0.0001 );
            eps_b0 = max( eps_b0 * .999985, 0.0001 );
            eps_b3 = max( eps_b3 * .999985, 0.0001 );

            weightcost  = 0.0002;
            dW32 = mom*dW32 + eps_w*( dSynapse32 - weightcost*W32 );
            dW01 = mom*dW01 + eps_w*( dSynapse01 - weightcost*W01   -     pd0'*sparsegrads1   / batchSize );
            dW12 = mom*dW12 + eps_w*( dSynapse12 - weightcost*W12   - pd1_pos'*sparsegrads2   / batchSize ...
                                                                    -(pd2_pos'*sparsegrads1)' / batchSize );
            
            db0 = mom*db0 + eps_b0*dAct0;
            db1 = mom*db1 + eps_b3*dAct1 - eps_b3*mean(sparsegrads1);
            db2 = mom*db2 + eps_b3*dAct2 - eps_b3*mean(sparsegrads2);
            db3 = mom*db3 + eps_b3*dAct3; % ??? b0 or b3 ???
            
            W01 = W01 + dW01;
            W12 = W12 + dW12;
            W32 = W32 + dW32;
            
            b0 = b0 + db0;
            b1 = b1 + db1;
            b2 = b2 + db2;
            b3 = b3 + db3;          
        end
        fprintf(1, ' Av. neuron recon. error: %1.3f, ', cum_err / nBatches );
        fprintf(1, '# TRAINING fails (of 60k): %5d \n', 60000 - nCorrect );
    end
    
    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
    
    function [dSynapse01, dSynapse12, dSynapse32, dAct0, dAct1, dAct2, dAct3, avNeuronErr] = ...
    getDeltas( data, targets )
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        s3 = targets;
        pd0 = data;
        
        B0 = repmat(b0, batchSize, 1);
        B1 = repmat(b1, batchSize, 1);
        B2 = repmat(b2, batchSize, 1);
        B3 = repmat(b3, batchSize, 1);

        % MEAN FIELD: Fix s0 (sample) and s3 (target) and bounce between pd1 and pd2
        % ELoDBM(S/H) Fig 6
        % Note pd1 has to compensate for lack of initial input from L2 by doubling up on input from L0
        pd1 = sigmoid( pd0*2*W01 + B1 ); % ??? Error in orig, failed to do 2*b1rep or b1rep+hidrecbiases
        pd2 = sigmoid( pd1*  W12 + B2 + s3*W32 );

        for meanFieldUpdate = 1:10
            pd1 = sigmoid( pd0*W01 + B1 + pd2*W12' );        
            pd2 = sigmoid( pd1*W12 + B2 +  s3*W32  ); % Error in orig. ?
        end
        % ^ pd0 not s0 (in 3 places above), see last sentence before 3.3 in S/H ELPDBM
        
        s0 = binarySampleFrom( pd0 );
        
        % Av. Synapse Activity
        A01_pos = mean_prod(  s0, pd1 );
        A12_pos = mean_prod( pd1, pd2 );
        A32_pos = mean_prod(  s3, pd2 );

        % Av. Neuron Activity
        A0_pos = mean(  s0 );
        A1_pos = mean( pd1 );
        A2_pos = mean( pd2 );
        A3_pos = mean(  s3 );

        pd0_pos = pd0;
        pd1_pos = pd1;
        pd2_pos = pd2;
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        pd0_neg = sigmoid( B0 + pd1*W01' ); % ???  
        
        expZ = exp( pd2*W32' + B3 );
        pd3 = expZ ./ repmat( sum(expZ,2), 1, neurons3 );
        s3 = binarySampleFromPDs( pd3 );
        
        % Make a guess!
        [~, index_pd  ] = max( pd3    , [], 2 );
        [~, index_targ] = max( targets, [], 2 );      
        nCorrect = nCorrect + nnz( index_pd == index_targ );
            
        %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for iter = 1:5
            % First, to update pd2 we need samples from L1 and L3
            s1 = binarySampleFrom( pd1 );
            
            % get s2
            pd2 = sigmoid( s1*W12 + B2 + s3*W32 );
            s2 = binarySampleFrom( pd2 );

            % get s3
            expZ = exp( s2*W32' + B3 );
            pd3 = expZ ./ repmat( sum(expZ,2), 1, neurons3 );
            s3 = binarySampleFromPDs( pd3 );           
            
            % get pd1
            pd0 = sigmoid( s1*W01' + B0 );
            s0 = binarySampleFrom( pd0 );
            
            pd1 = sigmoid( s0*W01 + B1 + s2*W12' );
        end

        A01_neg = mean_prod(  s0, pd1 );
        A12_neg = mean_prod( pd1, pd2 );
        A32_neg = mean_prod(  s3, pd2 );

        A0_neg = mean( s0 );
        A1_neg = mean( pd1 );
        A2_neg = mean( pd2 );
        A3_neg = mean( s3 );


        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        avNeuronErr = mean( rms(pd0_pos-pd0_neg,2) );

        % synapse activity delta (i.e. Discrepancy) between positive and negative phases
        dSynapse01 = A01_pos - A01_neg;
        dSynapse12 = A12_pos - A12_neg;
        dSynapse32 = A32_pos - A32_neg;
        
        % similarly, delta for activities(states) of neurons in layers 1,2,3
        dAct0 = A0_pos - A0_neg;
        dAct1 = A1_pos - A1_neg;
        dAct2 = A2_pos - A2_neg;
        dAct3 = A3_pos - A3_neg;
        
        % DISPLAY
        if ~rem(batchesDone,50)
            figure(1);
            imageGrid( s0', 28, 28 );
        end
            
    end
end