% Fix W01 and pretrain W12 as well as W23 and biases b1 b2 b3
function [W12, W32, b1, b2, b3] = ...
        pretrainL2( maxepoch, W01, neurons1, neurons2, ...
                    batchdata_train, batchtargets_train, ...
                    batchdata_test , batchtargets_test )
        
    eps_W12_initial    = 0.05;   % Learning rate for weights
    eps_b1_initial     = 0.05;   % Learning rate for biases of visible units
    eps_b2_initial     = 0.05;   % Learning rate for biases of hidden units
    
    weightcost  = 0.001;
    momInitial  = 0.5;
    momFinal    = 0.9;
    
    % (SL_ELoDBM Algo 1.2 or SH_ELP4DBM Algo 3.2) Freeze W1
    
    
    [batchSize, neurons3, nBatches] = size( batchtargets_train ); % ~ = num pixels

    % Initialize symmetric weights and biases    
    W12 = 0.01*randn(neurons1, neurons2);  dW12 = 0*W12;
    W32 = 0.01*randn(neurons3, neurons2);  dW32 = 0*W32;
    
    b1 = zeros( 1, neurons1);  db1 = 0*b1;
    b2 = zeros( 1, neurons2);  db2 = 0*b2;
    b3 = zeros( 1, neurons3);  db3 = 0*b3;
    
    
    for epoch = 1 : maxepoch
        CD = ceil( epoch / 20 );
        
        fprintf(1,'epoch %d (CD %d) ', epoch, CD);
        
        eps_w12 = eps_W12_initial / CD;
        eps_b1  = eps_b1_initial  / CD;
        eps_b2  = eps_b2_initial  / CD;
        
        avNeuronErr_tot = 0;
        
        batchesDone = 0;
        for randomBatch = randperm( nBatches )
            batchesDone = batchesDone + 1;
            
            if ~mod(batchesDone,20), fprintf(1,'.'); end
                        
            data    =    batchdata_train( :, :, randomBatch );
            targets = batchtargets_train( :, :, randomBatch );
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [dSynapse12, dSynapse32, dAct1, dAct2, dAct3, avNeuronErr] ...
                = getDeltas( data, targets );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            avNeuronErr_tot = avNeuronErr_tot + avNeuronErr;
            
            % UPDATE WEIGHTS AND BIASES
            mom = (epoch<=5)*momInitial + (epoch>5)*momFinal;
            
            dW12 = mom*dW12 + eps_w12 * ( dSynapse12 - weightcost * W12 );  W12 = W12 + dW12;
            dW32 = mom*dW32 + eps_w12 * ( dSynapse32 - weightcost * W32 );  W32 = W32 + dW32;
            
            db1 = mom*db1 + eps_b1 * dAct1;  b1 = b1 + db1;
            db2 = mom*db2 + eps_b2 * dAct2;  b2 = b2 + db2;
            db3 = mom*db3 + eps_b1 * dAct3;  b3 = b3 + db3;
        end
        
        fprintf(1, ' error %1.3f, ', avNeuronErr_tot / nBatches);
        
        % Look at the test scores
        %if rem(epoch,10) == 0
            err = testerr( batchdata_test, batchtargets_test, W01, b1, W12, b1, b2, W32, b3 );
            fprintf( 1, 'TEST fails (of 10k): %d \n', err );
        %end
        
    end
    
    
    function [dSynapse12, dSynapse32, dAct1, dAct2, dAct3, avNeuronErr] ...
            = getDeltas( data, targets )
        B1 = repmat( b1, batchSize, 1);
        B2 = repmat( b2, batchSize, 1);
        B3 = repmat( b3, batchSize, 1);
            
        % - - - - -  POSITIVE PHASE  - - - - -
        
        % Clamp L0 (visible layer) to data and get sample from L1 (See ELoDBM Fig 2 & Algo 1.2)
        
        s3 = targets;
        
        pd0 = data;
        s0 = binarySampleFrom( pd0 );
                    
        pd1 = s0*2*W01 + B1;
        s1 = binarySampleFrom( pd1 ); % <-- ELP4BDM Algo 3.2
        s1_pos = s1;
        
        % Clamping L1 (to this sample) AND L3 (output layer) to target,
        %  get probability distribution for L2 (second hidden layer).
        pd2 = sigmoid( s1*W12 + B2 + s3*W32 );
        
        A12_clampingL1 = mean_prod( s1, pd2 );
        A32_clampingL1 = mean_prod( s3, pd2 );
        
        A1_clampingL1 = mean(  s1 );
        A2_clampingL1 = mean( pd2 ); % notice we use p.d. not discrete sample
        A3_clampingL1 = mean(  s3 );
        
        % - - - - -  NEGATIVE PHASE  - - - - -
        
        % Bounce between L2 and L1 a few times updating p.d. for each
        for cd = 1 : CD
            % Sample from L2's p.d. ...
            s2 = binarySampleFrom( pd2 );
            
            % Sample from L3 (SoftMax -- only ONE label turns on):
            expZ = exp( s2 * W32'  +  B3 );
            pd3 = expZ ./ repmat( sum(expZ,2), 1, neurons3 );          
            s3 = binarySampleFromPDs( pd3 );
            
            % ... and update p.d. for L2.
            pd1 = sigmoid( s2*2*W12' + B1 ); % Get sample from L1 (as if L0 didn't exist)
            s1 = binarySampleFrom( pd1 );
            
            pd2 = sigmoid( s1*W12 + B2 + s3*W32 );
        end
        
        A12_thresh = mean_prod( s1, pd2 );
        A32_thresh = mean_prod( s3, pd2 );
        
        A1_thresh = mean(  s1 );
        A2_thresh = mean( pd2 ); % p.d. 'MeanField Reconstruction'
        A3_thresh = mean(  s3 );
        
        s1_neg = s1;
        
        % - - - - - - O U T P U T S - - - - - -
        
        avNeuronErr = mean( rms(s1_pos-s1_neg,2) );
        
        % synapse activity delta (i.e. Discrepancy) between positive and negative phases
        dSynapse12 = A12_clampingL1 - A12_thresh;
        dSynapse32 = A32_clampingL1 - A32_thresh;
        
        % similarly, delta for activities(states) of neurons in layers 1,2,3
        dAct1 = A1_clampingL1 - A1_thresh;
        dAct2 = A2_clampingL1 - A2_thresh;
        dAct3 = A3_clampingL1 - A3_thresh;
    end
    
end

function [err] = testerr( ...
            batchdata, batchtargets, W01, b1_, ...
            W12, b1, b2, W32, b3 )

    [batchSize, ~, nBatches] = size(batchdata);
    counter = zeros(10,10000);

    range_ = @(b) (b-1)*batchSize + ( 1 : batchSize );
    
    % targets_all = zeros(10000,10);
    % for b = 1 : nBatches
    %     targets_all( range_(b), :) = batchtargets(:,:,b);
    % end
    
    % http://uk.mathworks.com/matlabcentral/answers/3643-efficiently-converting-a-3d-matrix-to-a-2d-matrix
    tmp = permute( batchtargets, [1,3,2] );
    targets_all = reshape( tmp, [], size( batchtargets, 2 ) );

    B1_ = repmat( b1_, batchSize, 1 );
    B2  = repmat( b2 , batchSize, 1 );

    for b = 1 : nBatches
        inter = zeros( batchSize, 10 );
        data = batchdata( :, :, b );

        pd1 = sigmoid( data*2*W01 + B1_ );

        for d = 1 : 10
            targets = zeros( batchSize, 10 );  targets(:,d) = 1;

            z2 = pd1*W12 + B2 + targets*W32;
            
            % Shortcut:
            %   We want the most probable softmax unit, so we can ignore the softmax denom/partition
            %   So we want the highest exp(z3), and as exp is monotonic, the highest z3 will do
            %   and z3 = sum( sigma(z2_1) + sigma(z2_2) + ... )
            % grief I don't have a clue. wtf.
            p_vl  = pd1*b1' + sum( log(1+exp(z2)), 2 ) + targets*b3';
            inter(:,d) = p_vl;
        end

        counter(:, range_(b)) = inter';
    end

    [~ , J ] = max(    counter', [], 2 );
    [~, J1 ] = max( targets_all, [], 2 );
    
    err = length( find(J~=J1) );
end
