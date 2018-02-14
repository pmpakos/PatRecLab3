function [C,predictions] = bayes_class(data,avg_all,bayes_var,a_priori)
    cnt = 0;
    
    ind=find(data(:,1)==-1);
    data(ind,1)=2;
    
    for i=1:1:size(data,1)
        curr_digit = data(i,2:size(data,2));
        for k=1:1:2
            likelihood(k) = mvnpdf(curr_digit,avg_all(k,:),bayes_var(:,:,k));
            a_posteriori(k,:) = likelihood(k) .* a_priori(k);
        end
        [~,index] = max(a_posteriori);
        predictions(i) = index;
        if(index == data(i,1))
            cnt = cnt + 1;
        end
    end
    C = confusionmat(data(:,1),predictions);
end