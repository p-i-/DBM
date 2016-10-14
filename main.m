function main( bypassToStage )
    clc
    %dbquit all
    close all
    dbstop if error
    warning on verbose 
    [homeFolder,~,~] = fileparts(which('main'));  cd( homeFolder );
        
    % Add paths for source (.m) files
    pathlist = { '.', './src', './src/util', './src/functions' };
    for k = pathlist
        p = fullfile( pwd, k{1} );
        fprintf( '\tAdding path ''%s'' \n', p );
        addpath(p);
    end;
    
    %bypassToStage = '1'; % during debug, I override this (remove for final version)
    % Depending on parameter setting, load in previously saved values
    % 0 will recreate everything from scratch
    % 4 (default) bypasses all but the last (backprop) stage
    if (~exist('bypassToStage', 'var')), bypassToStage = '0'; end
    bypassToStage = bypassToStage - '0';
    if bypassToStage < 1, 
        if( exist( './data/', 'dir') ),  rmdir( './data/', 's' );  end
        mkdir( './data/' );
        mkdir( './data/out/' ); % foo 
    end 
    warning off MATLAB:DELETE:FileNotFound
    if bypassToStage < 2, delete( './data/out/pretrain1.mat' ); end
    if bypassToStage < 3, delete( './data/out/pretrain2.mat' ); end
    if bypassToStage < 4, delete( './data/out/meanField.mat' ); end
    clear bypassToStage
    warning on MATLAB:DELETE:FileNotFound    

    batchSize = 100;
    
    neurons0 = 28*28;
    neurons1 = 12;
    neurons2 = 13;
    
    maxepoch= 8; % <-- test
    
    close all
    
    % Load MNIST data
    fprintf(1, '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - \n');    
    fprintf(1, 'Loading MNIST data \n' );
    
    MNIST_src_path      = './data/MNIST_src/';
    MNIST_unpacked_path = './data/MNIST_unpacked/';
    
    if exist( [MNIST_src_path 't10k-labels-idx1-ubyte'], 'file') 
        fprintf(1, 'Skipping\n' );
    else
        download_raw_MNIST( MNIST_src_path );
    end

    
    fprintf(1, '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - \n');    
    fprintf(1, 'Unpacking \n' );
    
    if exist( [MNIST_unpacked_path 'train9.mat'], 'file' ) 
        fprintf(1, 'Skipping\n' );
    else
        unpack_MNIST( MNIST_src_path, MNIST_unpacked_path );
    end
    
    % PRETRAIN Level 1 (unsupervised, so don't need labels)
    fprintf(1, '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - \n');
    fprintf(1, 'PRE-TRAINING Level 1 with RBM: %d-%d \n', neurons0, neurons1);
    
    % Promote variableNotFound warning to error
    %   http://undocumentedmatlab.com/blog/trapping-warnings-efficiently
    warning('error', 'MATLAB:load:variableNotFound'); %#ok<CTPCT> % 
    try
        load data/out/pretrain1.mat  W01 b0 b1
        fprintf(1, 'Skipping\n' );
    catch
        [batchdata_train, ~, ~, ~] = makebatches( MNIST_unpacked_path, batchSize );
        
        [W01, b0, b1] = pretrainL1( maxepoch, neurons1, batchdata_train );
        
        save data/out/pretrain1.mat  W01 b0 b1       
    end
    
    
    % PRETRAIN Level 2
    fprintf(1, '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - \n');    
    fprintf(1, 'PRE-TRAINING Layer 2 with RBM: %d-%d \n', neurons1, neurons2);
    
    try
        load data/out/pretrain2.mat  W12 W32 b1 b2 b3
        fprintf(1, 'Skipping\n' );
    catch
        [ batchdata_train, batchtargets_train, ...
          batchdata_test , batchtargets_test ] = makebatches( MNIST_unpacked_path, batchSize );

        %b1_old = b1;
        
        [W12, W32, b1, b2, b3] = pretrainL2( ...
            maxepoch, W01, neurons1, neurons2, ...
            batchdata_train, batchtargets_train, ...
            batchdata_test , batchtargets_test );
        
        save data/out/pretrain2.mat  W12 W32 b1 b2 b3       
    end

    
    % DBM
    fprintf(1, '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - \n');
    fprintf(1, 'Learning a (two-layer) Deep Bolztamnn Machine: \n');
    
    try
        load data/out/meanField.mat  W01 W12 W32 b0 b1 b2 b3
        fprintf(1, 'Skipping\n' );
    catch
        [batchdata_train, batchtargets_train, ~ , ~] = makebatches( MNIST_unpacked_path, batchSize );

        dbm_mf( maxepoch, W01, W12, W32, b0, b1, b2, b3, ...
            batchdata_train, batchtargets_train )

        save data/out/meanField.mat  W01 W12 W32 b0 b1 b2 b3       
    end
    
    % Backprop to fine-tune
    fprintf(1, '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - \n');
    fprintf(1, 'Fine-tuning (using BackProp) for classification: \n');
    
    [ batchdata_train, batchtargets_train, ...
      batchdata_test , batchtargets_test ] = makebatches( MNIST_unpacked_path, batchSize );
    
    backprop( maxepoch, batchdata_train, batchtargets_train, ...
                        batchdata_test , batchtargets_test, ...
                        W01, W12, W32, b1, b2, b3 );
end