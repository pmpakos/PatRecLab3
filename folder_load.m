function ret =  folder_load(basepath)
    directory = dir(basepath); % contains everything, along with . and .. (first two elements)
    directory(1)=[];directory(1)=[]; % get rid of them
    del_fields = {'folder','date','bytes','isdir','datenum'}; 
    ret = rmfield(directory,del_fields); % get rid of unnecessary fields
end