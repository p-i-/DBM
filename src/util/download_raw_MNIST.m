function download_raw_MNIST( dest_path )
    %fprintf(1, 'Downloading raw MNIST data:\n');
    
    % If the data has already been unpacked we can return.
    % if exist( [dest_path 'train9.mat'], 'file'), return; end    
    
    %if ~exist(src_path , 'dir'), mkdir(src_path ); end
    %if ~exist(tmp_path , 'dir'), mkdir(tmp_path ); end
    if ~exist(dest_path, 'dir'), mkdir(dest_path); end
    
    train_img = 'train-images-idx3-ubyte'; train_labels = 'train-labels-idx1-ubyte';
    test_img  = 't10k-images-idx3-ubyte' ; test_labels  = 't10k-labels-idx1-ubyte';
        
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    % Locate or download raw MNIST data   
    for cell_ = { train_img, train_labels, test_img, test_labels }
        f_name = cell_{ 1 };
        f_path = [dest_path f_name];
        if ~exist( f_path, 'file' ),
            url_ = ['http://yann.lecun.com/exdb/mnist/' f_name '.gz'];
            gz = [f_path '.gz'];
            try
                fprintf( 1, ['Downloading ' url_ ' ... '] );
                websave( gz, url_ );
                
                fprintf( 1, ' Unzipping...' );
                gunzip( gz, dest_path );
                delete( gz );
            catch
                error 'Failed!'
            end
            fprintf( 1, ' Done! \n' );
        end
    end
        
    fprintf( 1, [ 'Found:'    ...
        '\n  * ' train_img    ...
        '\n  * ' train_labels ...
        '\n  * ' test_img     ...
        '\n  * ' test_labels  ...
        '\n in ' dest_path '\n' ] );
end