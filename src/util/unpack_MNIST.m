% pi@pipad.org (20 Sep 2016)

function unpack_MNIST( src_path, dest_path )
    %if exist( [dest_path 
    
    tmp_path = [dest_path 'tmp/'];
    
    if ~exist(tmp_path , 'dir'), mkdir(tmp_path ); end
    
    fprintf( 1, ' Unpacking...\n');
    
    train_img = 'train-images-idx3-ubyte'; train_labels = 'train-labels-idx1-ubyte';
    test_img  = 't10k-images-idx3-ubyte' ; test_labels  = 't10k-labels-idx1-ubyte';

    unpack( train_img, train_labels, 1000, 'train', 60 );
    unpack( test_img ,  test_labels, 1000,  'test', 10 );
    
    %close all
    rmdir( tmp_path, 's' ) % remove files AND folder
    return
    
    % = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =  
    
    function unpack( imagesFile, labelsFile, nChunkSize, strDataSet, nBatches )
        tmpFilename = @(k) [tmp_path strDataSet num2str(k) '.ascii'];

        % open source files
        hImagesFile = fopen( [src_path imagesFile], 'r' );
        hLabelsFile = fopen( [src_path labelsFile], 'r' );

        if hImagesFile==-1 || hLabelsFile==-1,
            error( 'convert() failed to open a source file' );
        end

        % trim header bumpf??
        [~, ~] = fread( hImagesFile, 4, 'int32' );
        [~, ~] = fread( hLabelsFile, 2, 'int32' );

        % create tempfile for each digit class (e.g. `./data/tmp/test_5.ascii`)
        hAsciiTempFile = cell(1,10);
        for d = 0 : 9,
            hAsciiTempFile{d+1} = fopen( tmpFilename(d), 'w' );
        end

        % Write e.g. all `5` images into `./tmpAscii/test_5.ascii` etc.
        fprintf(1, ['Unpacking MNIST images [dataset: ' strDataSet '] (prints 10 dots) \n']);
        for i = 1 : nBatches,
            fprintf('.');
            rawimages = fread( hImagesFile, 28*28*nChunkSize, 'uchar' );
            rawlabels = fread( hLabelsFile,       nChunkSize, 'uchar' );

            rawimages = reshape( rawimages, 28*28, nChunkSize );

            for j = 1 : nChunkSize,
                fprintf( hAsciiTempFile{rawlabels(j)+1}, '%3d ', rawimages(:,j) );
                fprintf( hAsciiTempFile{rawlabels(j)+1}, '\n' );
            end
        end

        % Convert temp .ascii to .mat
        fprintf(1, '\n');
        for d=0:9,    
            D = load( tmpFilename(d), '-ascii' );
            fprintf(1, 'Saving %5d digits of class %d\n', size(D,1), d);

            save( [dest_path strDataSet num2str(d) '.mat'], 'D', '-mat' );
        end;

        fclose('all');
    end
end
