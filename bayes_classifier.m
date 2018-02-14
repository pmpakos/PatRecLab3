function [accuracy,precision,recall,f1_score] = bayes_classifier(curr_train,curr_test)
%     curr_ind1 = find(curr_ind~= k_fold);
%     curr_ind2 = find(curr_ind== k_fold);
%     curr_train = curr_data(curr_ind1,:);
%     curr_eval = curr_data(curr_ind2,:);

    %following will be used in naive bayes classifier 
    %two classes 1 and -1 (-1 is always the second to come)
    a_priori1(1) = sum(curr_train(:,1)==1)/size(curr_train,1);
    a_priori1(2) = sum(curr_train(:,1)==-1)/size(curr_train,1);
    avg_all = avg_calc(curr_train);

    diagvar = var_calc(curr_train);
    [C,~] = bayes_class(curr_test,avg_all,diagvar,a_priori1);

    TP = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TN = C(2,2);

    accuracy = (TP+TN)/(TP+FP+FN+TN);
    precision = TP/(TP+FN);
    recall = TP/(TP+FN);
    f1_score = precision*recall/(precision+recall);

end
