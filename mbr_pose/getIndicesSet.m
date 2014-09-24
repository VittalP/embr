function indicesSet = getIndicesSet(numTestExamples, n)

siz = floor(numTestExamples/n);
indicesSet = cell(1,n);
for i=1:n-1;
    indicesSet{i} = [(i-1)*siz+1:i*siz];
end
indicesSet{n} = [(n-1)*siz+1:numTestExamples];