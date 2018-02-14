function [avg,cnt] = avg_calc(data)
    avg = zeros([2,size(data,2)-1]);
    cnt = zeros([1,2]);
    
    ind=find(data(:,1)==-1);
    data(ind,1)=2;

    
    for k=1:1:2
        sum = zeros([1,size(data,2)-1]);
        for i=1:1:size(data,1)
            if(data(i,1)==k)
                cnt(k) = cnt(k)+1;
                curr_digit = data(i,2:size(data,2));
                for j=1:1:size(curr_digit,2)
                    sum(j) = sum(j) + curr_digit(j);
                end
            end
        end
        avg(k,:) = sum/cnt(k);
    end
end
