function [new_train, new_test] = SplTrainTestData(data,pct)
    randind = randperm(size(data,1)); % take a random permutation of data so as to split them
    rand_data = data(randind,:);
    % split data into 1-pct % train data and pct % test data
    [Train, Test] = crossvalind('HoldOut', rand_data(:,1), pct); 
    new_train = rand_data(Train==1,:);
    new_test = rand_data(Test==1,:);
end