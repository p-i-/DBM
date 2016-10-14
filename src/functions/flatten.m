function [vecT,sizes] ...
  = flatten( mats )

    % Note: Could do: reshape = @(X) X(:)';  theta = arrayfun( reshape, W01, W21, W23, W32, b1, b2, b3 );

    sizes = zeros(2,numel(mats));
    % build the size vector:
    for k = 1:numel(mats)
        sizes(:,k) = size(mats{k}) .';
    end
    L = prod(sizes);
    vec = zeros(1,sum(L));
    offsets = cumsum([0 L]);
    % flattening all matrices to one vector
    for k = 1:numel(mats)
        vec( offsets(k)+1 : offsets(k+1) ) =  mats{k}(:) .';
    end
    
    vecT = vec .';
end
