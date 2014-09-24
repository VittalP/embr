function validationIndices = getValidationIndices(indicesSet, testIndex)

validationIndices = [];
for i=1:length(indicesSet)
    if(i == testIndex)
        continue;
    end
    validationIndices = [validationIndices, indicesSet{i}];
end