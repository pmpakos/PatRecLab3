function [accuracy,precision,recall,f1_score] = kNN_classifier(curr_train,curr_test,flag,curr_ind,k_fold,k_nn)
    if (flag==0) % used for hyperparameter evaluation
        curr_ind1 = find(curr_ind~= k_fold);
        curr_ind2 = find(curr_ind== k_fold);
        curr_eval = curr_train(curr_ind2,:); %evaluation set
        curr_train = curr_train(curr_ind1,:);
        [C,~] = K_Nearest_Neighbors(k_nn,curr_train,curr_eval);
    else
        [C,~] = K_Nearest_Neighbors(k_nn,curr_train,curr_test);
    end
    TP = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TN = C(2,2);

    accuracy = (TP+TN)/(TP+FP+FN+TN);
    precision = TP/(TP+FN);
    recall = TP/(TP+FN);
    f1_score = precision*recall/(precision+recall);
        
end