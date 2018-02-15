function [C,prediction] = K_Nearest_Neighbors(k,train,test)        
%%%%%%% simple kNN - yes!!!
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

%%%%%%% weighted kNN - maybe not!!!
% %     compute pairwise distances between each test instance vs. all training data
%     D = pdist2(test, train, 'euclidean');
%     [D,idx] = sort(D,2);
%     trainClass = train(:,1);
%     trainClass(trainClass==-1)=2;
%     %     K nearest neighbors
%     D = D(:,1:k);
%     idx = idx(:,1:k);
%     %     majority vote
%     votes=zeros(size(test,1),10);
%     for i=1:1:size(D,1)
%         for j=1:1:k
%             if k==1
%                 w(i,j) = 1;
%             else
%                 w(i,j)=(((D(i,k)-D(i,j))/(D(i,k)-D(i,1))))*((D(i,k)+D(i,1))/(D(i,k)+D(i,j)));
%             end
%             votes(i,trainClass(idx(i,j)))=votes(i,trainClass(idx(i,j)))+w(i,j);
%         end    
%         votes(i,:)=votes(i,:)/sum(votes(i,:));
%         [~,prediction(i)]=max(votes(i,:));
%     end
%     prediction(prediction==2)=-1;
% 
%     %   performance (confusion matrix and success percentage)
%     testClass = test(:,1);
%     C = confusionmat(testClass, prediction);
end