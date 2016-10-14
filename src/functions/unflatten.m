function cell_of_mat ...
  = unflatten( vecT, sizes )

    vec = vecT .';
    numMats = size(sizes,2);
    cell_of_mat = cell(numMats ,1);
    offsets = cumsum([0 prod(sizes)]);
    % initialize all matrices to correct size and fill with values
    for k = 1:numel(cell_of_mat)
        cell_of_mat{k} = zeros( sizes(:,k)' );
        cell_of_mat{k}(:) = vec( offsets(k)+1 : offsets(k+1) );
    end
end
