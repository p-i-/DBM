function [pixelGrid] = imageGrid(imstack,srcW,srcH,border_width,nRows,hflip,vflip)
    [~,N] = size(imstack); % images in batch/stack
    
    if(nargin<3), srcH = srcW;           end
    if(nargin<4), border_width= 2;       end
    if(nargin<5), nRows = ceil(sqrt(N)); end
    if(nargin<6), hflip = false;         end
    if(nargin<7), vflip = false;         end


    dstW = srcW+border_width;
    dstH = srcH+border_width;

    nCols = ceil(N/nRows);
    pixelGrid = min(imstack(:)) + zeros( nRows*dstW, nCols*dstH );

    r=0; c=1;
    for k=1:N
        r=r+1;
        if r > nRows,  r = 1; c = c+1;  end
        
        if(hflip), image = reshape( imstack(:,k), [], srcW )';
        else       image = reshape( imstack(:,k), [], srcH ) ; 
        end

        pixelGrid( (r-1)*dstW  + (1:srcW), ...
                   (c-1)*dstH  + (1:srcH) )  = image';

    end

    if(vflip), pixelGrid = flipud( pixelGrid ); end

    imagesc( pixelGrid );  colormap gray;  axis equal;  axis off;
    
    drawnow;
end