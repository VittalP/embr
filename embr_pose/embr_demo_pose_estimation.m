%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to perform approx Min Bayes Risk (MBR) prediction on the Pose
% Estimation problem with DivMBest solutions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% matlabpool open;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paths
% in globals
embr_divmbest_globals;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters

name = 'PARSE'; params.name = name;
% --------------------
% specify model parameters
% number of mixtures for 26 parts
K = [6 6 6 6 6 6 6 6 6 6 6 6 6 6 ...
    6 6 6 6 6 6 6 6 6 6 6 6]; params.K = K;
% Tree structure for 26 parts: pa(i) is the parent of part i
% This structure is implicity assumed during data preparation
% (PARSE_data.m) and evaluation (PARSE_eval_pcp)
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25]; params.pa = pa;
% Spatial resolution of HOG cell, interms of pixel width and hieght
% The PARSE dataset contains low-res people, so we use low-res parts
sbin = 4; params.sbin = sbin;


%% EMBR parameters

% type = 'perturb';
type = 'divmbest';
params.type = type;
nummodes = 50; params.nummodes = nummodes;

if(strcmp(type, 'divmbest'))
    lrange = -[0.001:0.002:0.2];
    lrange = lrange(1:25); % DivMBest
    %lrange = lrange(randperm(length(lrange)));
end

if(strcmp(type, 'perturb'))
    lrange = [];
    lrange = [0.1:0.5:10];
    lrange = [lrange, 1:10:101];
    lrange = [lrange, 0.001:0.02:0.51];
    lrange = lrange(randperm(length(lrange)));
    lrange = sort(unique(lrange));
end

one_scale = 0; params.one_scale = one_scale;
arange = 0;
Trange = [0:0.05:1];

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare training and testing images and part bounding boxes
% You will need to write custom *_data() functions for your own dataset
[pos, neg, test] = PARSE_data(name);
pos = point2box(pos,pa);
params.test = test;
numTestExamples = length(test);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training
%model = trainmodel(name,pos,neg,K,pa,sbin);
% load existing model for now
load('PARSE_model.mat');
params.model = model;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing phase 1
% human detection + pose estimation
suffix = num2str(K')';
model.thresh = min(model.thresh,-2);
boxes = testmodel(name,model,test,suffix);
det = PARSE_transback(boxes);
params.suffix = suffix;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluation 1: average precision of keypoints
% You will need to write your own APK evaluation code for your data structure
[meanapk_deva apk] = PARSE_eval_apk(det, test, 0.1);
%[meanapk_mmodes apk_mmodes] = PARSE_eval_apk(det_mmodes, test, 0.1);

%% Perform cross validation
nfold = 10;
display(['Performing ' num2str(nfold) ' fold cross validation.']);

indicesSet = getIndicesSet(numTestExamples, nfold);

% loop through the folds
all_mbr_test_all_folds_fileName = ['./cache/' type '_all_mbr_test_all_folds.mat'];
if(exist(all_mbr_test_all_folds_fileName, 'file'))
    load(all_mbr_test_all_folds_fileName);
else
    for testIndex = 1:length(indicesSet)
        display(['Testing on test index ' num2str(testIndex) '/' num2str(nfold)]);
        
        validationIndices = getValidationIndices(indicesSet, testIndex);
        testIndices = indicesSet{testIndex};
        
        %% Grid search over the validation set
        all_mbrs_val_fileName = ['./cache/' type '_all_mbrs_' num2str(nfold) 'fold_set' num2str(testIndex) '.mat'];
        if(exist(all_mbrs_val_fileName, 'file'))
            load(all_mbrs_val_fileName);
        else
            all_mbrs_val = zeros(length(lrange), length(Trange), nummodes);
            for pl=1:length(lrange)
                lambda = lrange(pl)
		params.lambda = lambda;                
                % Detect the various modes
		output = DivMBest_pose_estimation(params);
		boxes_mmodes = output.boxes_mmodes;
		det_mmodes = output.det_mmodes;

                sol_mpck = zeros(nummodes,numTestExamples);
                sol_iou = zeros(nummodes,numTestExamples);
                
                % compute accuracy of all modes
                sol_mpck_sol_iou_fileName = ['./cache/' type '_sol_mpck_pck_sol_iou' num2str(lambda) '_.mat'];
                if(exist(sol_mpck_sol_iou_fileName,'file'))
                    load(sol_mpck_sol_iou_fileName );
                else
                    for pim=1:numTestExamples
                        for ps=1:nummodes
                            detm.point = det_mmodes(pim).point(:,:,ps);
                            detm.score = det_mmodes(pim).score(ps);
                            
                            [sol_mpck(ps,pim), pck, sol_iou(ps,pim)] = PARSE_eval_pck(detm, test(pim), 0.1, 0);
                        end
                    end
                    save(sol_mpck_sol_iou_fileName, 'sol_mpck', 'pck', 'sol_iou', '-v7.3');
                end
                
                %%
                % call to mbr function that constructs pairwise eval_pck matrix over boxes
                % multiplies by exp(Score) and takes min
                % compute pairwise loss
                
                solpairacc_fileName = ['./cache/' type '_solpairacc_' num2str(lambda) '.mat'];
                if(exist(solpairacc_fileName, 'file'))
                    load(solpairacc_fileName);
                else
                    fprintf('SolPairAcc: ');
                    solpairacc = cell(1,numTestExamples);
                    parfor pim=1:numTestExamples
                        if mod(pim,10)==0
                            fprintf('%d...',pim);
                        end
                        % compute m x m loss matrix
                        [mpck, pck, iou] = computePairwiseLoss(det_mmodes,nummodes, pim);
                        solpairacc{pim} = mpck/2 + mpck'/2;
                    end
                    save(solpairacc_fileName, 'solpairacc');
                    fprintf('\n');
                end
                %%
                % make mbr prediction
                
                % Create new variables pertaining to just the validation set
                boxes_mmodes_val = boxes_mmodes(validationIndices);
                det_mmodes_val = det_mmodes(validationIndices);
                test_val = test(validationIndices);
                solpairacc_val = solpairacc(validationIndices);
                
                all_mbrs_lambda_val_fileName = ['./cache/' type '_all_mbrs_' num2str(lambda) '_' num2str(nfold) 'fold_set' num2str(testIndex) '.mat'];
                if(exist(all_mbrs_lambda_val_fileName, 'file'))
                    load(all_mbrs_lambda_val_fileName);
                else
                    all_mbrs_lambda = zeros(length(Trange), nummodes);
                    parfor pt = 1:length(Trange)
                        T = Trange(pt);
                        fprintf('MBR: pl=%d/%d T=%d/%d\n', pl, length(lrange), pt, length(Trange));
                        all_mbrs_lambda(pt,:) = mbr_prediction_wrapper(T, nummodes, boxes_mmodes_val, det_mmodes_val, solpairacc_val, test_val);
                        fprintf('\n');
                    end
                    save(all_mbrs_lambda_val_fileName, 'all_mbrs_lambda');
                end
                
                all_mbrs_val(pl,:,:) = all_mbrs_lambda;
            end
            save(all_mbrs_val_fileName, 'all_mbrs_val');
        end
        
        %% Test on held out set
        display('Testing on the held out set...');
        test_set_results_fn = ['./cache/' type '_oracle_test_results_' num2str(nfold) 'fold_testSet' num2str(testIndex) '.mat'];
        if(exist(test_set_results_fn, 'file'))
            load(test_set_results_fn);
        else
            for ii = 1:5
                display(['Case: ' num2str(ii)]);
                
                %Get the best parameters
                switch ii
                    case 1
                        %validate over all 3
                        %lambda_m*, T_m*, m*
                        [pl,pt,ps] = ind2sub(size(all_mbrs_val), find(all_mbrs_val == max(all_mbrs_val(:))));
                    case 2
                        % validate over 2 parameters
                        %lambda_m*, T_inf, m*
                        pt = length(Trange); % Set T = inf
                        all_mbrs_val_2 = squeeze(all_mbrs_val(:,pt,:));
                        [pl,ps] = ind2sub(size(all_mbrs_val_2), find(all_mbrs_val_2 == max(all_mbrs_val_2(:))));
                    case 3
                        % validate over 1 parameter
                        %lambda_M, T_inf, M
                        pt = length(Trange); % Set T = inf
                        ps = 50; % Choose a particular M
                        all_mbrs_val_3 = squeeze(all_mbrs_val(:,pt,ps));
                        [pl] = ind2sub(size(all_mbrs_val_3), find(all_mbrs_val_3 == max(all_mbrs_val_3(:))));
                    case 4
                        % validate over 2 parameters
                        %lambda_M, T_M, M
                        ps = 50;
                        all_mbrs_val_4 = squeeze(all_mbrs_val(:,:,ps));
                        [pl,pt] = ind2sub(size(all_mbrs_val_4), find(all_mbrs_val_4 == max(all_mbrs_val_4(:))));
                    case 5
                        % validate over 2 parameters for each m
                        %lambda_m, T_m, m
                        for ps = 1:nummodes
                            all_mbrs_val_5 = squeeze(all_mbrs_val(:,:,ps));
                            [pl,pt] = ind2sub(size(all_mbrs_val_5), find(all_mbrs_val_5 == max(all_mbrs_val_5(:))));
                            pl_5(ps) = pl(1); % There might be more than one lambda producing the maximum score
                            pt_5(ps) = pt(1); % There might be more than one temperature producing the maximum score
                            all_mbr_val_best(ps) = squeeze(all_mbrs_val(pl(1), pt(1), ps));
                        end
                end
                
                % Collect the best parameters
                M = ps(1);
                lambda = lrange(pl(1));
                T = Trange(pt(1));
                
                if( ii ~= 5)
                    all_mbr_val_best = squeeze(all_mbrs_val(pl(1), pt(1), :));
                end
                % Load the precomputed results for the learnt best parameters
                load(['./cache/' type '_solpairacc_' num2str(lambda) '.mat']);
                load(['./cache/' type '_sol_mpck_pck_sol_iou' num2str(lambda) '_.mat']);
                
		params.lambda = lambda;
		output = DivMBest_pose_estimation(params); % This just loads boxes_mmodes
		boxes_mmodes = output.boxes_mmodes;
		det_mmodes = params.det_mmodes;

                
                % Create new variables pertaining to just the test set
                boxes_mmodes_test = boxes_mmodes(testIndices);
                det_mmodes_test = det_mmodes(testIndices);
                sol_mpck_test = sol_mpck(1:nummodes,testIndices);
                solpairacc_test = solpairacc(testIndices);
                
                if(ii == 1) % Find oracle only for the three parameter validation case
                    fprintf('Oracle: ');
                    for ps=1:nummodes
                        fprintf('%d...',ps);
                        [spck, sind] = max(sol_mpck_test(1:ps,:),[],1);
                        
                        % get oracle detections
                        for pim=1:length(testIndices)
                            det_oracle_test(pim).point = det_mmodes_test(pim).point(:,:,sind(1,pim));
                            det_oracle_test(pim).score = det_mmodes_test(pim).score(sind(1,pim));
                        end
                        
                        oracle_mapk_test(ps) = PARSE_eval_apk(det_oracle_test, test(testIndices), 0.1, 0);
                    end
                    case_struct.oracle_mapk_test = oracle_mapk_test;
                    fprintf('\n');
                end
                
                fprintf('MBR: ');
                for ps = 1:nummodes
                    fprintf('%d...',ps);
                    
                    if( ii == 5 ) % Corresponds to validating over lambda and temperature for each mode
                        lambda = lrange(pl_5(ps));
                        T = Trange(pt_5(ps));
                        
                        load(['./cache/' type '_solpairacc_' num2str(lambda) '.mat']);
                        load(['./cache/' type '_sol_mpck_pck_sol_iou' num2str(lambda) '_.mat']);
                        
                        boxes_mmodes = testmodel_mmodes(name,model,test,suffix,nummodes,one_scale,lambda,type); % This just loads boxes_mmodes
                        det_mmodes = PARSE_transback(boxes_mmodes);
                        
                        % Create new variables pertaining to just the test set
                        boxes_mmodes_test = boxes_mmodes(testIndices);
                        det_mmodes_test = det_mmodes(testIndices);
                        sol_mpck_test = sol_mpck(1:nummodes,testIndices);
                        solpairacc_test = solpairacc(testIndices);
                    end
                    
                    [mbr, mbrind, mbr_boxes] = mbr_prediction(boxes_mmodes_test, det_mmodes_test, solpairacc_test ,length(testIndices), ps, T);
                    det_mbr_test = PARSE_transback(mbr_boxes);
                    all_mbr_test(ps) = PARSE_eval_apk(det_mbr_test, test(testIndices), 0.1, 0);
                end
                
                fprintf('\n');
                if(ii == 5)
                    case_struct.pl_5 = pl_5;
                    case_struct.pt_5 = pt_5;
                else
                    case_struct.lambda = lambda;
                end
                case_struct.T = T;
                case_struct.M = M;
                case_struct.all_mbr_test = squeeze(all_mbr_test);
                case_struct.all_mbr_val_best = all_mbr_val_best;
                
                switch ii
                    case 1
                        %lambda_m*, T_m*, m*
                        case_struct.mbr(testIndex) = case_struct.all_mbr_test(case_struct.M);
                        case1 = case_struct;
                    case 2
                        %lambda_m*, T_inf, m*
                        case_struct.mbr(testIndex) = case_struct.all_mbr_test(case_struct.M);
                        case2 = case_struct;
                    case 3
                        %lambda_M, T_M, M
                        case3 = case_struct;
                    case 4
                        %lambda_M, T_inf, M
                        case4 = case_struct;
                    case 5
                        % validate over 2 parameters for each m
                        %lambda_m, T_m, m
                        case5 = case_struct;
                end
            end
            save(test_set_results_fn, 'oracle_mapk_test', 'case1', 'case2', 'case3', 'case4', 'case5');
        end
        
        for ii = 1:5
            switch ii
                case 1
                    %lambda_m*, T_m*, m*
                    all_folds_case1.all_mbr_test_all(testIndex,:) = case1.all_mbr_test;
                    all_folds_case1.oracle_mapk_test_all_folds(testIndex,:) = case1.oracle_mapk_test;
                    all_folds_case1.point(testIndex) = case1.all_mbr_test(case1.M);
                    all_folds_case1.T(testIndex) = case1.T;
                    all_folds_case1.M(testIndex) = case1.M;
                    all_folds_case1.mbr = case1.mbr;
                case 2
                    %lambda_m*, T_inf, m*
                    all_folds_case2.all_mbr_test(testIndex,:) = case2.all_mbr_test;
                    all_folds_case2.T(testIndex) = case2.T;
                    all_folds_case2.M(testIndex) = case2.M;
                    all_folds_case2.mbr = case2.mbr;
                case 3
                    %lambda_M, T_inf, M
                    all_folds_case3.all_mbr_test(testIndex,:) = case3.all_mbr_test;
                case 4
                    %lambda_M, T_M, M
                    all_folds_case4.all_mbr_test(testIndex,:) = case4.all_mbr_test;
                case 5
                    % validate over 2 parameters for each m
                    %lambda_m, T_m, m
                    all_folds_case5.all_mbr_test(testIndex,:) = case5.all_mbr_test;
            end
        end
    end
    save(all_mbr_test_all_folds_fileName, 'all_folds_case1', 'all_folds_case2', 'all_folds_case3', 'all_folds_case4', 'all_folds_case5');
end
%% Plot the results

lw = 4;
fsize = 30;

if(strcmp(type, 'perturb'))
    type = 'Perturb&MAP';
end
if(strcmp(type, 'divmbest'))
    type = 'DivMBest';
end

mean_oracle_mapk = mean(all_folds_case1.oracle_mapk_test_all_folds,1);

figure, hold on;
plot(mean_oracle_mapk*100,'k--s', 'linewidth',lw);
legendinfo{1} = 'Oracle';
plot(repmat(meanapk_deva*100,1,nummodes),'r-.', 'linewidth',lw);
legendinfo{2} = 'MAP';
count = 2;
for ii = 3:5  % Do not plot the three parameter validation curve
    switch ii
%         If you want to plot the three parameter validation curve,
%         uncomment the following case and loop through from 1:5 (above).
%         
        case 1
%             %lambda_m*, T_m*, m*
%             test_mbr_iou = case1.test_mbr_iou;
%             [m,ind] = max(test_mbr_iou);
%             c = 'b';            
%             leg = ['EMBR-' type ' (\lambda_{m*}=' num2str(case1.lambda) ', T_{m*}=' num2str(case1.T) ', m* = ' num2str(ind) ')'];
        case 2
            %lambda_m*, T_inf, m*
            mean_test_mbr_mapk = mean(all_folds_case2.all_mbr_test,1);
            [m,ind] = max(mean_test_mbr_mapk);
            c = [0.17, 0.51, 0.34];
            leg = ['EMBR-' type ' (\lambda_{m*}, T_{\infty}, m*)'];
        case 3
            %lambda_M, T_inf, M
            mean_test_mbr_mapk = mean(all_folds_case3.all_mbr_test,1);
            [m,ind] = max(mean_test_mbr_mapk);
            c = 'm';
            leg = ['EMBR-' type ' (\lambda_{M}, T_{\infty}, M)'];
        case 4
            %lambda_M, T_M, M
            mean_test_mbr_mapk = mean(all_folds_case4.all_mbr_test,1);
            [m,ind] = max(mean_test_mbr_mapk);
            c = 'b';
            leg = ['EMBR-' type ' (\lambda_{M}, T_{M}, M)'];
        case 5
            % validate over 2 parameters for each m
            %lambda_m, T_m, m
            
            mean_test_mbr_mapk = mean(all_folds_case5.all_mbr_test,1);
            [m,ind] = max(mean_test_mbr_mapk);
            c = [0.68, 0.47, 0];
            leg = ['EMBR-' type ' (\lambda_{m}, T_{m})'];            
    end
    plot(mean_test_mbr_mapk*100, '-s', 'Color', c, 'linewidth',lw);
    count = count+1;
    legendinfo{count} = leg;
end

% Dummy plots
plot(repmat(60.5,1,nummodes),'w', 'linewidth',lw);
str_mapk = num2str(mean(all_folds_case1.mbr)*100);
str_mapk = str_mapk(1:5);
leg = ['EMBR-' type ' (\lambda_{m*}, T_{m*}, m*) = ' str_mapk '%'];
count = count+1;
legendinfo{count} = leg;

plot(repmat(60.5,1,nummodes),'w', 'linewidth',lw);
str_mapk = num2str(mean(all_folds_case2.mbr)*100);
str_mapk = str_mapk(1:5);
leg = ['EMBR-' type ' (\lambda_{m*}, T_{\infty}, m*) = ' str_mapk '%'];
count = count+1;
legendinfo{count} = leg;

xlabel('# Solutions M', 'FontSize', fsize);
ylabel('Mean APK(%)', 'FontSize', fsize);
legend(legendinfo);
set(gcf, 'Color', 'w');
set(gca,'fontsize',fsize,'linewidth',lw);
