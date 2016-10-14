function activation = sigmoid( E ) % Energy
    activation = 1 ./ (1 + exp(-E));
end

