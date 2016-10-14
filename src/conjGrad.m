function [f, df] ...
  = conjGrad( theta, sizes, data, target, pd2_temp, do_init )
    
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
