clear;close all;clc;warning off;
tic
% %% PRELAB
% %% Step 1 - Pre-process Data
% basepath = './PRcourse_Lab3_data/MusicFileSamples/';
% music_contents = folder_load(basepath);
% 
% for i=1:1:size(music_contents,1)
%     cname = ['file',num2str(i),'.wav'];
%     full_name = strcat(basepath,cname);
%     [stereo,Fs] = audioread(full_name);
%     
%     stereo = stereo(1:2:end, :) ; % Subsampling by a factor of 0.5  
%     stereo = 0.5 * ( stereo(:, 1) + stereo(:, 2) ) ; % Stereo to Mono
%     stereo = int8((2^7 - 1) * stereo / max(abs(stereo))) ; % 8 bits / sample
%     beatles{i} = stereo; 
% end
% 
% %% Step 2 - Per Labeler Statistics
% basepath = './PRcourse_Lab3_data/EmotionLabellingData/';
% lab_file = folder_load(basepath);
% for j=1:1:size(lab_file,1)
%     cname = lab_file(j).name;
%     full_name = strcat(basepath,cname);
%     curr_labeler = load(full_name);
%     curr_labeler = curr_labeler.labelList;
%     Labelers(:,j) = curr_labeler;
%     
%     % a) mean and b) standard deviation calculation
%     mean_act(j) = mean(cell2mat({curr_labeler.activation}));
%     std_act(j) = std(cell2mat({curr_labeler.activation}));
%     mean_val(j) = mean(cell2mat({curr_labeler.valence}));
%     std_val(j) = std(cell2mat({curr_labeler.valence}));
%     
%     % c) co-occurence matrix & histogram
%     temp_co_occur = zeros(5,5);
%     for k=1:1:size(curr_labeler,1)
%         curr_act = curr_labeler(k).activation;
%         curr_val = curr_labeler(k).valence;
%         temp_co_occur(curr_act,curr_val) = temp_co_occur(curr_act,curr_val) + 1;   
%     end
%     co_occur{j} = temp_co_occur;
%     
%     figure;imagesc(temp_co_occur);
%     colormap('copper');colorbar;
%     xlabel('Valence');ylabel('Activation');
%     title(['Histogram of Labels for Labeler ',num2str(j)]);
%     title1 = ['histogram_',num2str(j),'.png'];
%     saveas(gcf,title1);
% end
% 
% %% Step 3 - Observed Agreement
% combos = [1 2; 1 3; 2 3];
% 
% for k=1:1:size(combos,1)
%     x = combos(k,1);
%     y = combos(k,2);
%     
%     ObsAgr_activ = abs(cell2mat({Labelers(:,x).activation}) - cell2mat({Labelers(:,y).activation}));
%     ObsAgr_val   = abs(cell2mat({Labelers(:,x).valence}) - cell2mat({Labelers(:,y).valence}));
%     
%     TotObsAgr_activ(k) = 1-mean(ObsAgr_activ)/4;
%     TotObsAgr_val(k) = 1-mean(ObsAgr_val)/4;
%     disp(['Observed Agreement ( activation ) between Labeler',num2str(x),' and Labeler',num2str(y),' is ',num2str(TotObsAgr_activ(k))]);
%     disp(['Observed Agreement (   valence  ) between Labeler',num2str(x),' and Labeler',num2str(y),' is ',num2str(TotObsAgr_val(k))]);
%   
%     histo_activ = zeros(5,1);
%     histo_val = zeros(5,1);
%     for j=1:1:size(ObsAgr_activ,2)
%         histo_activ( ObsAgr_activ(j) + 1) = histo_activ( ObsAgr_activ(j) + 1) +1;
%         histo_val( ObsAgr_val(j) + 1) = histo_val( ObsAgr_val(j) + 1) +1;
%     end
%     figure;histogram(ObsAgr_activ,0:4);
%     title(['Observed Agreement(activation) Labelers ',num2str(x),' & ',num2str(y)]);
%     title1 = ['ObsAgr_activ_',num2str(x),num2str(y),'.png'];
%     saveas(gcf,title1);
%     figure;histogram(ObsAgr_val,0:4);
%     title(['Observed Agreement(valence) Labelers ',num2str(x),' & ',num2str(y)]);
%     title1 = ['ObsAgr_val_',num2str(x),num2str(y),'.png'];
%     saveas(gcf,title1);
% end
% 
% %% Step 4 - Krippendorff's Alpha ( code obtained from https://goo.gl/PjXHkf )
% activ_labels = [cell2mat({Labelers(:,1).activation}) ; cell2mat({Labelers(:,2).activation}) ; cell2mat({Labelers(:,3).activation})];
% kalpha_activ = kriAlpha(activ_labels,'ordinal');
% 
% val_labels = [cell2mat({Labelers(:,1).valence}) ; cell2mat({Labelers(:,2).valence}) ; cell2mat({Labelers(:,3).valence})];
% kalpha_val = kriAlpha(val_labels,'ordinal');
% 
% %% Step 5 - Mean labels
% activation = mean(activ_labels);
% valence = mean(val_labels);
% 
% vals = [1, 1.3333, 1.6667, 2, 2.3333, 2.6667, 3, 3.3333, 3.6667, 4, 4.3333, 4.6667, 5];
% co_occurence = zeros(13,13);
% for j=1:1:size(activation,2)
%     curr_act1 = round(activation(j),4);
%     curr_act = find(vals==curr_act1);
%     curr_val1 = round(valence(j),4);
%     curr_val = find(vals==curr_val1);
%     co_occurence(curr_act,curr_val) = co_occurence(curr_act,curr_val) + 1;   
% end
% figure;imagesc(vals,vals,co_occurence);
% set(gca,'XTick',vals);set(gca,'YTick',vals);xtickangle(90);
% colormap('copper');colorbar;
% xlabel('Valence');ylabel('Activation');
% title('Histogram of Labels (averaged)');
% saveas(gcf,'histogram_avg.png');
% 
% %% Step 6 - MIR Toolbox Feature Extraction
% addpath(genpath('./MIRtoolbox1.7'))
% % basepath = './PRcourse_Lab3_data/MonoSamples/';
% % music_contents = folder_load(basepath);
% 
% for i=1:1:size(beatles,2)
%     i
%     curr_track = mirframe(beatles{i});
%     
%     % 1) Auditory Roughness
%     roughness = mirgetdata(mirroughness(curr_track));
%     rough_mean(1) = mean(roughness); 
%     rough_std = std(roughness);
%     
%     rough_med_pl = roughness;
%     rough_med_pl(rough_med_pl < median(rough_med_pl)) = [];
%     rough_mean(2) = mean(rough_med_pl); 
%     
%     rough_med_mi = roughness;
%     rough_med_mi(rough_med_mi > median(rough_med_mi)) = [];
%     rough_mean(3) = mean(rough_med_mi); 
%     
%     % 2) Rythmic Periodicity Along Auditory Channels
%     fluctuation = mirgetdata(mirsum(mirfluctuation(curr_track)));
%     fluct_max = max(fluctuation);
%     fluct_mean = mean(fluctuation);
%     
%     % 3) Key Clarity
%     key_mean = mean(mirgetdata(mirkey(curr_track)));
% 
%     % 4) Modality
%     mode = mirgetdata(mirmode(curr_track));
%     modality = mean(mode);
% 
%     % 5) Spectral Novelty
%     snov = mirgetdata(mirnovelty(curr_track));
%     sp_nov_mean = mean(snov);
%     
%     % 6) Harmonic Change Detection Function (HCDF)
%     hcdf = mirgetdata(mirhcdf(curr_track));
%     hcdf_mean = mean(hcdf);
%     
%     features(i).rough_mean1 = rough_mean(1);
%     features(i).rough_mean2 = rough_mean(2);
%     features(i).rough_mean3 = rough_mean(3);
%     features(i).rough_std = rough_std;
%     features(i).fluct_max = fluct_max;
%     features(i).fluct_mean = fluct_mean;
%     features(i).key_mean = key_mean;
%     features(i).modality = modality;
%     features(i).sp_nov_mean = sp_nov_mean;
%     features(i).hcdf_mean = hcdf_mean;
%     clc;
% end 
% 
% %% Step 7 - MFCC Extraction
% for i=1:1:size(beatles,2)
%     i
%     curr_track = mirframe(beatles{i}, 0.025, 's' ,0.01, 's');
%     
%     mfcc = mirgetdata(mirmfcc(curr_track, 'Bands', 26, 'Rank', 1:13))';
%     mfcc_delta = mirgetdata(mirmfcc(curr_track, 'Bands', 26, 'Rank', 1:13, 'Delta', 1))';
%     mfcc_deltadelta = mirgetdata(mirmfcc(curr_track, 'Bands', 26, 'Rank', 1:13, 'Delta', 2))';
%     
%     mfcc_mean = mean(mfcc);
%     mfcc_delta_mean = mean(mfcc_delta);
%     mfcc_deltadelta_mean = mean(mfcc_deltadelta);
% 
%     mfcc_std = std(mfcc);
%     mfcc_delta_std = std(mfcc_delta);
%     mfcc_deltadelta_std = std(mfcc_deltadelta);
%     
%     mfcc = sort(mfcc);
%     mfcc_delta = sort(mfcc_delta,2);
%     mfcc_deltadelta  = sort(mfcc_deltadelta,2);
%     
%     perc = round(0.1*size(mfcc,1));
%     perc2 = round(0.1*size(mfcc_delta,1));
%     perc3 = round(0.1*size(mfcc_deltadelta,1));
%     
%     mfcc_mean_low = mean(mfcc(1:perc,:));
%     mfcc_delta_mean_low = mean(mfcc_delta(1:perc2,:));
%     mfcc_deltadelta_mean_low = mean(mfcc_deltadelta(1:perc3,:));
% 
%     mfcc_mean_high = mean(mfcc(end-perc:end,:));
%     mfcc_delta_mean_high = mean(mfcc_delta(end-perc2:end,:));
%     mfcc_deltadelta_mean_high = mean(mfcc_deltadelta(end-perc3:end,:));
%     
%     mfcc_related.mfcc_mean = mfcc_mean;
%     mfcc_related.mfcc_delta_mean = mfcc_delta_mean;
%     mfcc_related.mfcc_deltadelta_mean = mfcc_deltadelta_mean;
% 
%     mfcc_related.mfcc_std = mfcc_std;
%     mfcc_related.mfcc_delta_std = mfcc_delta_std;
%     mfcc_related.mfcc_deltadelta_std = mfcc_deltadelta_std;
% 
%     mfcc_related.mfcc_mean_low = mfcc_mean_low;
%     mfcc_related.mfcc_delta_mean_low = mfcc_delta_mean_low;
%     mfcc_related.mfcc_deltadelta_mean_low = mfcc_deltadelta_mean_low;
% 
%     mfcc_related.mfcc_mean_high = mfcc_mean_high;
%     mfcc_related.mfcc_delta_mean_high = mfcc_delta_mean_high;
%     mfcc_related.mfcc_deltadelta_mean_high = mfcc_deltadelta_mean_high;
% 
%     features(i).mfcc_related = mfcc_related;
%     clc;
% end

%% Step 8 - 1st lab classification algorithms
%% Step 9 - Weka Installation
%% LAB
load 'PRcourse_Lab3_data/matlab_final.mat'
%% Step 10 

act_remove = [];
val_remove = [];
for i=1:1:size(features,2)
    final_data1(i).activation = activation(i);
    final_data1(i).mir = [features(i).rough_mean1 features(i).rough_mean2 features(i).rough_mean3 features(i).rough_std features(i).fluct_max features(i).fluct_mean features(i).key_mean features(i).modality features(i).sp_nov_mean features(i).hcdf_mean];
    final_data1(i).mfcc = [ features(i).mfcc_related.mfcc_mean features(i).mfcc_related.mfcc_delta_mean features(i).mfcc_related.mfcc_deltadelta_mean features(i).mfcc_related.mfcc_std features(i).mfcc_related.mfcc_delta_std features(i).mfcc_related.mfcc_deltadelta_std features(i).mfcc_related.mfcc_mean_low features(i).mfcc_related.mfcc_delta_mean_low features(i).mfcc_related.mfcc_deltadelta_mean_low features(i).mfcc_related.mfcc_mean_high features(i).mfcc_related.mfcc_delta_mean_high features(i).mfcc_related.mfcc_deltadelta_mean_high];
    final_data1(i).combined = [final_data1(i).mir final_data1(i).mfcc];
    
    final_data2(i).valence = valence(i);
    final_data2(i).mir = [features(i).rough_mean1 features(i).rough_mean2 features(i).rough_mean3 features(i).rough_std features(i).fluct_max features(i).fluct_mean features(i).key_mean features(i).modality features(i).sp_nov_mean features(i).hcdf_mean];
    final_data2(i).mfcc = [ features(i).mfcc_related.mfcc_mean features(i).mfcc_related.mfcc_delta_mean features(i).mfcc_related.mfcc_deltadelta_mean features(i).mfcc_related.mfcc_std features(i).mfcc_related.mfcc_delta_std features(i).mfcc_related.mfcc_deltadelta_std features(i).mfcc_related.mfcc_mean_low features(i).mfcc_related.mfcc_delta_mean_low features(i).mfcc_related.mfcc_deltadelta_mean_low features(i).mfcc_related.mfcc_mean_high features(i).mfcc_related.mfcc_delta_mean_high features(i).mfcc_related.mfcc_deltadelta_mean_high];
    final_data2(i).combined = [final_data2(i).mir final_data2(i).mfcc];
     
    if (final_data1(i).activation > 3)
        final_data1(i).activation = 1;
    elseif (final_data1(i).activation < 3)
        final_data1(i).activation = -1;
    else
        act_remove = [act_remove i];
    end
    
    if (final_data2(i).valence> 3)
        final_data2(i).valence = 1;
    elseif (final_data2(i).valence < 3)
        final_data2(i).valence = -1;
    else
        val_remove = [val_remove i];
    end
end

final_data1(act_remove) = [];
final_data2(val_remove) = [];

for i=1:1:size(final_data1,2)
    mir1(i,:) = final_data1(i).mir;
    mfcc1(i,:) = final_data1(i).mfcc;
    comb1(i,:) = final_data1(i).combined;
end
mir1 = mapstd(mir1')';  %mir1 = mapminmax(mir1')'; % ws pros th 2h diastash gi auto to diplo '...
mfcc1= mapstd(mfcc1')'; %mfcc1= mapminmax(mfcc1')';
comb1= mapstd(comb1')'; %comb1= mapminmax(comb1')';
for i=1:1:size(final_data2,2)
    mir2(i,:) = final_data2(i).mir;
    mfcc2(i,:) = final_data2(i).mfcc;
    comb2(i,:) = final_data2(i).combined;
end
mir2 = mapstd(mir2')';  %mir2 = mapminmax(mir2')';
mfcc2= mapstd(mfcc2')'; %mfcc2= mapminmax(mfcc2')';
comb2= mapstd(comb2')'; %comb2= mapminmax(comb2')';

for i=1:1:size(final_data1,2)
    final_data1(i).mir = mir1(i,:);
    final_data1(i).mfcc = mfcc1(i,:);
    final_data1(i).combined = comb1(i,:);
end

for i=1:1:size(final_data2,2)
    final_data2(i).mir = mir2(i,:);
    final_data2(i).mfcc = mfcc2(i,:);
    final_data2(i).combined = comb2(i,:);
end

%% Step 11 - Features Preparation
mir1 = [] ; mfcc1 = [] ; comb1 = [];
mir2 = [] ; mfcc2 = [] ; comb2 = [];

for i=1:1:size(final_data1,2)
    mir1(i,:) = [final_data1(i).activation final_data1(i).mir];
    mfcc1(i,:) = [final_data1(i).activation final_data1(i).mfcc];
    comb1(i,:) = [final_data1(i).activation final_data1(i).combined];
end

for i=1:1:size(final_data2,2)
    mir2(i,:) = [final_data2(i).valence final_data2(i).mir];
    mfcc2(i,:) = [final_data2(i).valence final_data2(i).mfcc];
    comb2(i,:) = [final_data2(i).valence final_data2(i).combined];
end

%% Step 12 - kNN Classifier
accuracy_activ_mir = [] ; precision_activ_mir = [] ; recall_activ_mir = [] ; f1_score_activ_mir = []; accuracy_activ_mfcc = [] ; precision_activ_mfcc = [] ; recall_activ_mfcc = [] ; f1_score_activ_mfcc = []; accuracy_activ_comb = [] ; precision_activ_comb = [] ; recall_activ_comb = [] ; f1_score_activ_comb = [];
accuracy_val_mir = []   ; precision_val_mir = []   ; recall_val_mir = []   ; f1_score_val_mir = []  ; accuracy_val_mfcc = []   ; precision_val_mfcc = []   ; recall_val_mfcc = []   ; f1_score_val_mfcc = []  ; accuracy_val_comb = []   ; precision_val_comb = []   ; recall_val_comb = []   ; f1_score_val_comb = [];

cnt = 0;
for k_nn=1:2:13
    cnt=cnt+1;
    for k_fold=1:1:3
        %activation
        [mir1_train,mir1_test]= SplTrainTestData(mir1,0.2);
        [mfcc1_train,mfcc1_test]= SplTrainTestData(mfcc1,0.2);
        [comb1_train,comb1_test]= SplTrainTestData(comb1,0.2);

        %valence
        [mir2_train,mir2_test]= SplTrainTestData(mir2,0.2);
        [mfcc2_train,mfcc2_test]= SplTrainTestData(mfcc2,0.2);
        [comb2_train,comb2_test]= SplTrainTestData(comb2,0.2);
        
        %activation
        [accuracy_activ_mir(cnt,k_fold), precision_activ_mir(cnt,k_fold), recall_activ_mir(cnt,k_fold), f1_score_activ_mir(cnt,k_fold)]  = kNN_classifier(mir1_train, mir1_test, k_nn);
        [accuracy_activ_mfcc(cnt,k_fold),precision_activ_mfcc(cnt,k_fold),recall_activ_mfcc(cnt,k_fold),f1_score_activ_mfcc(cnt,k_fold)] = kNN_classifier(mfcc1_train,mfcc1_test,k_nn);
        [accuracy_activ_comb(cnt,k_fold),precision_activ_comb(cnt,k_fold),recall_activ_comb(cnt,k_fold),f1_score_activ_comb(cnt,k_fold)] = kNN_classifier(comb1_train,comb1_test,k_nn);

        %valence
        [accuracy_val_mir(cnt,k_fold), precision_val_mir(cnt,k_fold), recall_val_mir(cnt,k_fold), f1_score_val_mir(cnt,k_fold)]  = kNN_classifier(mir2_train, mir2_test, k_nn);
        [accuracy_val_mfcc(cnt,k_fold),precision_val_mfcc(cnt,k_fold),recall_val_mfcc(cnt,k_fold),f1_score_val_mfcc(cnt,k_fold)] = kNN_classifier(mfcc2_train,mfcc2_test,k_nn);
        [accuracy_val_comb(cnt,k_fold),precision_val_comb(cnt,k_fold),recall_val_comb(cnt,k_fold),f1_score_val_comb(cnt,k_fold)] = kNN_classifier(comb2_train,comb2_test,k_nn);
    end
end

acc_activ_mir_knn = mean(accuracy_activ_mir,2);   acc_activ_mfcc_knn = mean(accuracy_activ_mfcc,2);    acc_activ_comb_knn = mean(accuracy_activ_comb,2);
acc_val_mir_knn   = mean(accuracy_val_mir,2);     acc_val_mfcc_knn   = mean(accuracy_val_mfcc,2);      acc_val_comb_knn   = mean(accuracy_val_comb,2);
pre_activ_mir_knn = mean(precision_activ_mir,2);  pre_activ_mfcc_knn = mean(precision_activ_mfcc,2);   pre_activ_comb_knn = mean(precision_activ_comb,2);
pre_val_mir_knn   = mean(precision_val_mir,2);    pre_val_mfcc_knn   = mean(precision_val_mfcc,2);     pre_val_comb_knn   = mean(precision_val_comb,2);
rec_activ_mir_knn = mean(recall_activ_mir,2);     rec_activ_mfcc_knn = mean(recall_activ_mfcc,2);      rec_activ_comb_knn = mean(recall_activ_comb,2);
rec_val_mir_knn   = mean(recall_val_mir,2);       rec_val_mfcc_knn   = mean(recall_val_mfcc,2);        rec_val_comb_knn   = mean(recall_val_comb,2);
f1_sc_activ_mir_knn = mean(f1_score_activ_mir,2); f1_sc_activ_mfcc_knn = mean(f1_score_activ_mfcc,2);  f1_sc_activ_comb_knn = mean(f1_score_activ_comb,2);
f1_sc_val_mir_knn   = mean(f1_score_val_mir,2);   f1_sc_val_mfcc_knn   = mean(f1_score_val_mfcc,2);    f1_sc_val_comb_knn   = mean(f1_score_val_comb,2);


%% Step 13 - Bayes Classifier
accuracy_activ_mir = [] ; precision_activ_mir = [] ; recall_activ_mir = [] ; f1_score_activ_mir = []; accuracy_activ_mfcc = [] ; precision_activ_mfcc = [] ; recall_activ_mfcc = [] ; f1_score_activ_mfcc = []; accuracy_activ_comb = [] ; precision_activ_comb = [] ; recall_activ_comb = [] ; f1_score_activ_comb = [];
accuracy_val_mir = []   ; precision_val_mir = []   ; recall_val_mir = []   ; f1_score_val_mir = []  ; accuracy_val_mfcc = []   ; precision_val_mfcc = []   ; recall_val_mfcc = []   ; f1_score_val_mfcc = []  ; accuracy_val_comb = []   ; precision_val_comb = []   ; recall_val_comb = []   ; f1_score_val_comb = [];

for k_fold=1:1:3
    %activation
    [mir1_train,mir1_test]= SplTrainTestData(mir1,0.2);
    [mfcc1_train,mfcc1_test]= SplTrainTestData(mfcc1,0.2);
    [comb1_train,comb1_test]= SplTrainTestData(comb1,0.2);

    %valence
    [mir2_train,mir2_test]= SplTrainTestData(mir2,0.2);
    [mfcc2_train,mfcc2_test]= SplTrainTestData(mfcc2,0.2);
    [comb2_train,comb2_test]= SplTrainTestData(comb2,0.2);

    %activation
    [accuracy_activ_mir(k_fold), precision_activ_mir(k_fold), recall_activ_mir(k_fold), f1_score_activ_mir(k_fold) ] = bayes_classifier(mir1_train, mir1_test);
    [accuracy_activ_mfcc(k_fold),precision_activ_mfcc(k_fold),recall_activ_mfcc(k_fold),f1_score_activ_mfcc(k_fold)] = bayes_classifier(mfcc1_train,mfcc1_test);
    [accuracy_activ_comb(k_fold),precision_activ_comb(k_fold),recall_activ_comb(k_fold),f1_score_activ_comb(k_fold)] = bayes_classifier(comb1_train,comb1_test);
    
    %valence
    [accuracy_val_mir(k_fold), precision_val_mir(k_fold), recall_val_mir(k_fold), f1_score_val_mir(k_fold) ] = bayes_classifier(mir2_train, mir2_test);
    [accuracy_val_mfcc(k_fold),precision_val_mfcc(k_fold),recall_val_mfcc(k_fold),f1_score_val_mfcc(k_fold)] = bayes_classifier(mfcc2_train,mfcc2_test);
    [accuracy_val_comb(k_fold),precision_val_comb(k_fold),recall_val_comb(k_fold),f1_score_val_comb(k_fold)] = bayes_classifier(comb2_train,comb2_test);
end

acc_activ_mir_bayes = mean(accuracy_activ_mir);   acc_activ_mfcc_bayes = mean(accuracy_activ_mfcc);    acc_activ_comb_bayes = mean(accuracy_activ_comb);
acc_val_mir_bayes   = mean(accuracy_val_mir);     acc_val_mfcc_bayes   = mean(accuracy_val_mfcc);      acc_val_comb_bayes   = mean(accuracy_val_comb);
pre_activ_mir_bayes = mean(precision_activ_mir);  pre_activ_mfcc_bayes = mean(precision_activ_mfcc);   pre_activ_comb_bayes = mean(precision_activ_comb);
pre_val_mir_bayes   = mean(precision_val_mir);    pre_val_mfcc_bayes   = mean(precision_val_mfcc);     pre_val_comb_bayes   = mean(precision_val_comb);
rec_activ_mir_bayes = mean(recall_activ_mir);     rec_activ_mfcc_bayes = mean(recall_activ_mfcc);      rec_activ_comb_bayes = mean(recall_activ_comb);
rec_val_mir_bayes   = mean(recall_val_mir);       rec_val_mfcc_bayes   = mean(recall_val_mfcc);        rec_val_comb_bayes   = mean(recall_val_comb);
f1_sc_activ_mir_bayes = mean(f1_score_activ_mir); f1_sc_activ_mfcc_bayes = mean(f1_score_activ_mfcc);  f1_sc_activ_comb_bayes = mean(f1_score_activ_comb);
f1_sc_val_mir_bayes   = mean(f1_score_val_mir);   f1_sc_val_mfcc_bayes   = mean(f1_score_val_mfcc);    f1_sc_val_comb_bayes   = mean(f1_score_val_comb);

%% Step 14 - PCA
dims = [4, 18, 25, 40, 55];

dim=0;
for i=1:1:size(dims,2)
    dim=dim+1; 
    %activation
    pca_comb1 = [comb1(:,1) ppca(comb1(:,2:end)',dims(i))];

    %valence
    pca_comb2 = [comb2(:,1) ppca(comb2(:,2:end)',dims(i))];

    
    % PCA+kNN
    accuracy_activ_comb = [] ; precision_activ_comb = [] ; recall_activ_comb = [] ; f1_score_activ_comb = [];
    accuracy_val_comb = []   ; precision_val_comb = []   ; recall_val_comb = []   ; f1_score_val_comb = [];

    cnt = 0;
    for k_nn=1:2:13
        cnt=cnt+1;
        for k_fold=1:1:3
            %activation
            [comb1_train,comb1_test]= SplTrainTestData(pca_comb1,0.2);
            %valence
            [comb2_train,comb2_test]= SplTrainTestData(pca_comb2,0.2);

            %activation
            [accuracy_activ_comb(cnt,k_fold),precision_activ_comb(cnt,k_fold),recall_activ_comb(cnt,k_fold),f1_score_activ_comb(cnt,k_fold)] = kNN_classifier(comb1_train,comb1_test,k_nn);
            %valence
            [accuracy_val_comb(cnt,k_fold),precision_val_comb(cnt,k_fold),recall_val_comb(cnt,k_fold),f1_score_val_comb(cnt,k_fold)] = kNN_classifier(comb2_train,comb2_test,k_nn);
        end
    end
    
    acc_activ_comb_knn_pca(dim,:) = mean(accuracy_activ_comb,2);
    acc_val_comb_knn_pca(dim,:)   = mean(accuracy_val_comb,2);
    pre_activ_comb_knn_pca(dim,:) = mean(precision_activ_comb,2);
    pre_val_comb_knn_pca(dim,:)   = mean(precision_val_comb,2);
    rec_activ_comb_knn_pca(dim,:) = mean(recall_activ_comb,2);
    rec_val_comb_knn_pca(dim,:)   = mean(recall_val_comb,2);
    f1_sc_activ_comb_knn_pca(dim,:) = mean(f1_score_activ_comb,2);
    f1_sc_val_comb_knn_pca(dim,:)   = mean(f1_score_val_comb,2);

    % PCA+Bayes
    accuracy_activ_comb = [] ; precision_activ_comb = [] ; recall_activ_comb = [] ; f1_score_activ_comb = [];
    accuracy_val_comb = []   ; precision_val_comb = []   ; recall_val_comb = []   ; f1_score_val_comb = [];

    for k_fold=1:1:3
        %activation
        [comb1_train,comb1_test]= SplTrainTestData(pca_comb1,0.2);
        %valence
        [comb2_train,comb2_test]= SplTrainTestData(pca_comb2,0.2);

        %activation
        [accuracy_activ_comb(k_fold),precision_activ_comb(k_fold),recall_activ_comb(k_fold),f1_score_activ_comb(k_fold)] = bayes_classifier(comb1_train,comb1_test);
        %valence
        [accuracy_val_comb(k_fold),precision_val_comb(k_fold),recall_val_comb(k_fold),f1_score_val_comb(k_fold)] = bayes_classifier(comb2_train,comb2_test);
    end
    acc_activ_comb_bayes_pca(dim) = mean(accuracy_activ_comb);
    acc_val_comb_bayes_pca(dim)   = mean(accuracy_val_comb);
    pre_activ_comb_bayes_pca(dim) = mean(precision_activ_comb);
    pre_val_comb_bayes_pca(dim)   = mean(precision_val_comb);
    rec_activ_comb_bayes_pca(dim) = mean(recall_activ_comb);
    rec_val_comb_bayes_pca(dim)   = mean(recall_val_comb);
    f1_sc_activ_comb_bayes_pca(dim) = mean(f1_score_activ_comb);
    f1_sc_val_comb_bayes_pca(dim)   = mean(f1_score_val_comb);
end

%% Step 15 - Weka Data Preparation
%step6 data
fnames_mir  = fieldnames(features); fnames_mir {11} = []; fnames_mir = fnames_mir (~cellfun('isempty',fnames_mir)); %for mir features
WekaDataPrep('activation','MIR',mir1,fnames_mir);
WekaDataPrep('valence','MIR',mir2,fnames_mir);

%step7 data
fnames_mfcc = fieldnames(features(1).mfcc_related); %fnames_mfcc = repelem(fnames_mfcc,13); % for mfcc features
cnt=0;
for i=1:1:size(fnames_mfcc,1)
    cnt=cnt+1;
    for j=1:1:13
        temp{13*(cnt-1)+j} = [fnames_mfcc{i},'_',num2str(j)];
    end
end
fnames_mfcc = temp';
WekaDataPrep('activation','MFCC',mfcc1,fnames_mfcc);
WekaDataPrep('valence','MFCC',mfcc2,fnames_mfcc);

%step6+7 data
fnames_comb = [fnames_mir ;fnames_mfcc]; 
WekaDataPrep('activation','Combined',comb1,fnames_comb);
WekaDataPrep('valence','Combined',comb2,fnames_comb);


%% Step 16 - Weka Classification Algorithms
%% Step 17 - 
toc