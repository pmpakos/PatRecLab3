function WekaDataPrep(string1,string2,data,fnames)
    fid = fopen([string1,'_',string2,'_', 'Weka.arff'], 'wt' );
    fprintf(fid, ['@RELATION emotion_',string1,'\n']);

    for i=1:1:size(fnames,1)
        fprintf(fid, ['@ATTRIBUTE ',fnames{i,1},' NUMERIC\n']);
    end

    fprintf(fid, ['@ATTRIBUTE ',string1,' {-1, 1}\n']);
    fprintf(fid, '\n@DATA\n');

    for i=1:1:size(data,1)
        for j=2:1:size(data,2)
            fprintf(fid, [num2str(data(i,j)),',']);
        end
        fprintf(fid, [num2str(data(i,1)),'\n']);    
    end

    fclose(fid);
end