function [accuracy,precision,recall,f1_score] = kNN_classifier(curr_train,curr_test,k_nn)
    [C,~] = K_Nearest_Neighbors(k_nn,curr_train,curr_test);
    
    TP = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TN = C(2,2);

    accuracy = (TP+TN)/(TP+FP+FN+TN);
    precision = TP/(TP+FN);
    recall = TP/(TP+FN);
    f1_score = precision*recall/(precision+recall);
        
end