function ret = var_calc(data)  
    ind=find(data(:,1)==-1);
    data(ind,1)=2;
    
    for k=1:1:2
        ind = find(data(:,1)==k);
        X = data(ind,2:size(data,2));
        var_all(k,:) =var(X);
    end
    for k=1:1:2
        index = var_all(k,:)==0;
        var_all(k,index) = 0.001; % dokimasame me eps, me 0.001, me 0.0001
        ret(:,:,k) = diag(var_all(k,:));
    end
end