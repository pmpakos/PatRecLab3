function [C,prediction] = K_Nearest_Neighbors(k,train,test)        
% compute pairwise distances between each test instance vs. all training data
    D = pdist2(test, train, 'euclidean');
    [D,idx] = sort(D,2);

% K nearest neighbors
    D = D(:,1:k);
    idx = idx(:,1:k);

% majority vote
    trainClass = train(:,1);
    [prediction,~] = mode(trainClass(idx),2);
    
% performance (confusion matrix and success percentage)
    testClass = test(:,1);
    C = confusionmat(testClass, prediction);
end