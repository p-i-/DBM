function [W01, b0, b1] = ...
pretrainL1( maxepoch, neurons1, batchdata )
    eps_w      = 0.05;   % Learning rate for weights 
    eps_b0     = 0.05;   % Learning rate for biases of visible units 
    eps_b1     = 0.05;   % Learning rate for biases of hidden units 

    %CD=1;   
    weightCost = 0.001;   
    momInitial = 0.5;
    momFinal   = 0.9;

    [batchSize, neurons0, nBatches] = size( batchdata );

    % Initializing symmetric weights and biases. 
    W01 = 0.001*randn(neurons0, neurons1);  dW01 = 0*W01;
    b0 =        zeros(1, neurons0)       ;  db0  = 0*b0;
    b1 =        zeros(1, neurons1)       ;  db1  = 0*b1;
    
tic
    for epoch = 1 : maxepoch
        fprintf(1,'epoch %d ',epoch);
        errsum=0;
        for batch = 1 : nBatches,
            if ~mod(batch,20), fprintf(1,'.'); end
            
            data = batchdata(:,:,batch);
            
            % For L0 states, L1 states and L0-L1 synapses calculate:
            %    [activity when data-vector clamped to L0] - [threshold activity (nothing clamped)]
            %    a.k.a. Positive phase - Negative phase
            % We use 1-step Contrastive Divergence (CD1)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [dSynapse01, dAct0, dAct1, av_err] = getDeltas( data );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % keep a tally of average per-neuron divergence of
            % reconstructions from original sampled vis-vector.
            errsum = errsum + av_err;
            
            % Fit W1 (,b0, b1) to data (Sal/LaR2010 Algo 1.1)
            % i.e. wire together synapses which fired together (hebbian learning)
            % and encourage neurons that fired a lot.
            mom = (epoch<=5)*momInitial + (epoch>5)*momFinal;

            dW01 = mom*dW01 + eps_w  * (dSynapse01 - weightCost*W01);
            db0  = mom*db0  + eps_b0 * dAct0;
            db1  = mom*db1  + eps_b1 * dAct1;

            W01 = W01 + dW01;
            b0  = b0  + db0;
            b1  = b1  + db1;
        end
        
        fprintf(1, ' Av Pixel Error %1.3f \r', errsum / nBatches);
    end
toc
    

    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    % Use 1-step CD
    function [dSynapse01, dAct0, dAct1, avPixelErr] ...
        = getDeltas( data )
    
        B0 = repmat( b0, batchSize, 1 );
        B1 = repmat( b1, batchSize, 1 );

        % Data is 28*28 greyscale pixels ranging 0.0 to 1.0
        % Consider each pixel as prob. dist. on layer 0 (L0)
        pd0 = data;
        
        % - - - - -  POSITIVE PHASE  - - - - -
        
        % convert each pixel to 0 or 1 accordingly to get VIS outputs.
        s0 = binarySampleFrom( pd0 );
        
        % Clamping L0 (visible layer) to s0, get L1 (hidden layer 1) activations, 
        % i.e. POSTERIOR pd (prob. dist.) for L1
        pd1 = sigmoid( s0*2*W01 + B1 ); % Eq. 7
        %                     ^ 2 copies of visible vector (Sal/LaR2010 Algo 1.1)
        
        % Get average 'firing-together-ness' for each L01 connection/synapse
        % Note we're using pd1 (float in [0,1] not s1 which would be 0 or 1.
        % 784 vis/inputs going to (say) 10 hiddens = a 784x10 v_i h_j matrix.
        % Sum over all batch items (say 100):  784x100 * 100x10 -> 784x10
        A01_clampingV = mean_prod( s0, pd1 ); % s0' * pd1  / batchSize;

        % Get expected activations for each visible and hidden neuron
        A0_clampingV = mean( s0 ); % <v_i h_j>_data in Eq. 5
        A1_clampingV = mean( pd1 );

        % - - - - -  NEGATIVE PHASE  - - - - -
        
        % Sample from our posterior dist. for L1 (hidden layer 1)
        s1 = binarySampleFrom( pd1 );            
        
        % Clamping L1 to this sample, get a POSTERIOR dist. for L0 (visible layer)
        pd0recon = sigmoid( s1*W01' + B0 ); % Eq. 8
        
        % from which we sample a 'Reconstruction' vis/data vector/sample ...
        s0recon = binarySampleFrom( pd0recon );
        
        % ... and throw it back thru hidden units to get a 'neg' HID dist.
        pd1recon = sigmoid( s0recon*2*W01 + B1 );

        % Imagine doing many + and - phases, collecting threshold/dreaming values for:
        %    - 'firing-together-ness' for each vis-hid connection/synapse
        %    - activation of each visible unit
        %    - activation of each hidden unit
        % ... then averaging over all batch items.
        A01_thresh  = mean_prod( s0recon, pd1recon ); % s0recon' * pd1recon  / batchSize;
        
        % why not use visReconPD & sampleFromVis?  see Section 3 of Training RBM.
        A0_thresh = mean( s0recon ); 
        A1_thresh = mean( pd1recon );    

        %  - - - - -  D I S P L A Y  - - - - - 
        
        if rem( batch, 600 ) == 0
            figure(1);
            imageGrid( s0recon', 28, 28 ); % Draw batch (as 10x10 ims)
            drawnow;
        end
        
        % - - - - - - O U T P U T S - - - - - -
        
        % Calculate how much of this synapse-activity is due to V
        dSynapse01 = A01_clampingV - A01_thresh;
        
        dAct0 = A0_clampingV - A0_thresh;
        dAct1 = A1_clampingV - A1_thresh;
        
        avPixelErr = mean( rms(s0recon-s0,2) );
    end
end

