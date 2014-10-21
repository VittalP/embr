%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to perform approx Min Bayes Risk (MBR) prediction on the 2-class
% interactive segmentation problem with DivMBest solutions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Paths
addpath(genpath('./utils'));
DivMBest_PATH = '~/divmbest/';
DivMBest_intseg_PATH = [DivMBest_PATH 'intseg/'];
addpath(genpath(DivMBest_intseg_PATH));

datadir = [DivMBest_intseg_PATH './voctest50data']; params.datadir = datadir;

%% Automatically download data if it does not exist
files = dir([datadir '/*.mat']);
if(length(files) ~= 50)
    try
        websave('voctest50data.tar', 'https://filebox.ece.vt.edu/~vittal/embr/voctest50data.tar');
        system(['tar -xvf voctest50data.tar']);
        system(['cp -rf voctest50data ' DivMBest_intseg_PATH]);
        system('rm -rf voctest50data');
    catch
        error('Unable to download/extract/move the PASCAL val data. Please do it manually.');
    end
end
    

gtdir = fullfile(datadir,'gtdir'); params.gtdir = gtdir;
savedir = './savedir'; params.savedir = savedir;

if ~exist(savedir,'dir');
    mkdir(savedir);
end

%% The variable 'type' is used to switch between the PnM and DivMBest cases.

%type = 'perturb'  % Comment this line and uncomment the following line if you want to use DivMBest
type = 'divMbest_boundary_'  % Comment this line and uncomment the above line if you want to use Perturb-And-MAP
params.type = type;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters

nummodes = 50; params.nummodes = nummodes;
nlabels = 2; params.nlabels = nlabels;

if(strcmp(type, 'divMbest_boundary_'))
    lrange = [0.1, 0.2, 0.3];
end

if(strcmp(type, 'perturb'))
    lrange = [0.01:0.005:0.05];
    lrange = [lrange, 0.1:0.05:0.5];
    lrange = [lrange, 0.6:0.1:1];
end
lrange = sort(unique(lrange));

Trange =  [10, 20, 50, 100, 200, 300, 500, 1000, 5000, 50000];

flist = dir(fullfile(datadir,'*.mat'));
ntest = length(flist);

%%
% load ground-truth
for pf = 1:ntest
    fname = flist(pf).name(1:end-4);
    gt{pf} = imread(sprintf('%s/%s.png',gtdir,fname));
end

validationIndices = 1:2:50; % Train on odd-numbered images
testIndices = 2:2:50; % Test on even numbered images
seg = cell(nummodes,ntest);
all_mbrs_fn = ['./savedir/' type 'all_mbrs_val.mat'];
if(exist(all_mbrs_fn, 'file'))
    load(all_mbrs_fn);
else
    for pl = 1:length(lrange)
        lambda = lrange(pl); params.lambda = lambda;
        seg_sol_iou_en_fn = ['./savedir/' type 'seg_sol_iou_en_' num2str(lambda) '.mat'];
        if(exist(seg_sol_iou_en_fn,'file'))
            load(seg_sol_iou_en_fn);
        else
            for pf = 1:ntest
                fname = flist(pf).name(1:end-4); params.fname = fname;
		params.gt = gt{pf};

		load_struct = load(sprintf('%s/%s.mat',datadir,fname));
		data_term = load_struct.data_term;
		labels = load_struct.labels; params.labels = labels;
		sparse_term = load_struct.sparse_term;
		
		%% ne
		nnodes = size(data_term,2);
		assert(nnodes==length(unique(labels)));
		ne = data_term;

		% DB:  swap 1 and 0 terms because something funny seems to be going on. Maybe Payman was maximizing. Or maybe these are outputs of classifiers (so scores, not energies)
		ne([1 2],:) = ne([2 1],:); params.ne = ne;

		% el
		[node1 node2 wt] = find(triu(sparse_term));
		nedges = length(wt);
		el = [node1 node2]'; params.el = el;

		% ee
		ee = zeros(4,nedges);
		ee(2,:) = wt;
		ee(3,:) = wt;
		params.ee = ee;


		%% Perform DivMbest
		output = DivMBest_intseg(params);

		seg(:,pf) = output.seg;
		sol_iou(:,pf) = output.sol_iou;
		sol_en(:,pf) = output.sol_en;
                
            end
            save(seg_sol_iou_en_fn, 'seg', 'sol_iou', 'sol_en', '-v7.3');
        end
        
        solpairacc_fn = ['./savedir/' type 'solpairacc_' num2str(lambda) '.mat'];
        if(exist(solpairacc_fn, 'file'))
            load(solpairacc_fn);
        else
            fprintf('SolPairAcc: ');
            parfor pf=1:ntest
                if mod(pf,10)==0
                    fprintf('%d...',pf);
                end
                % compute m x m loss matrix
                iou = [];
                for ps1 = 1:nummodes
                    for ps2 = 1:nummodes
                        [acc, precision, recall, iou(ps1,ps2), fmeasure] = computeStats(seg{ps1,pf}, seg{ps2,pf});
                    end
                end
                solpairacc{pf} = iou;
            end
            fprintf('\n');
            save(solpairacc_fn, 'solpairacc');
        end
        
        % make mbr prediction
        ntest1 = length(validationIndices);
        sol_en1 = sol_en(:,validationIndices);
        solpairacc1 = solpairacc(validationIndices);
        sol_iou1 = sol_iou(:, validationIndices);
        for pt = 1:length(Trange)
            T = Trange(pt);
            fprintf('MBR: ');
            for ps = 1:nummodes
                fprintf('%d...',ps);
                for pf=1:ntest1
                    prob = exp(-sol_en1(1:ps,pf)/T);
                    prob = prob/sum(prob);
                    
                    % because we have accuracy not loss
                    [mbr(pf) mbrind(pf)] = max(solpairacc1{pf}(1:ps,1:ps)*prob);
                end
                mbr_iou(ps) = mean(sol_iou1(sub2ind([nummodes ntest1], mbrind, 1:ntest1)));
            end
            all_mbrs(pl,pt,:) = mbr_iou;
            fprintf('\n');
        end
    end
    save(all_mbrs_fn, 'all_mbrs');
end

%% Test on held out set

display('Testing on the held out set');
test_set_results_fn = ['./savedir/' type '_oracle_test_results.mat'];
if(~exist(test_set_results_fn, 'file'))
    
    for ii = 1:5
        
        display(['Case: ' num2str(ii)]);
        
        %Get the best parameters
        
        switch ii
            case 1
                %validate over all 3
                %lambda_m*, T_m*, m*
                [pl,pt,ps] = ind2sub(size(all_mbrs), find(all_mbrs == max(all_mbrs(:))));
            case 2
                % validate over 2 parameters
                %lambda_m*, T_inf, m*
                pt = length(Trange); % Set T = inf
                all_mbrs_2 = squeeze(all_mbrs(:,pt,:));
                [pl,ps] = ind2sub(size(all_mbrs_2), find(all_mbrs_2 == max(all_mbrs_2(:))));
            case 3
                % validate over 1 parameter
                %lambda_M, T_inf, M
                pt = length(Trange); % Set T = inf
                ps = 50; % Choose a particular M
                all_mbrs_3 = squeeze(all_mbrs(:,pt,ps));
                [pl] = ind2sub(size(all_mbrs_3), find(all_mbrs_3 == max(all_mbrs_3(:))));
            case 4
                % validate over 2 parameters
                %lambda_M, T_M, M
                ps = 50; % Choose a particular M
                all_mbrs_4 = squeeze(all_mbrs(:,:,ps));
                [pl,pt] = ind2sub(size(all_mbrs_4), find(all_mbrs_4 == max(all_mbrs_4(:))));
            case 5
                % validate over 2 parameters for each m
                %lambda_m, T_m, m
                for ps = 1:nummodes
                    all_mbrs_5 = squeeze(all_mbrs(:,:,ps));
                    [pl,pt] = ind2sub(size(all_mbrs_5), find(all_mbrs_5 == max(all_mbrs_5(:))));
                    pl_5(ps) = pl(1); % Sometimes, multiple lambdas produce the best performance. So, choose the first one.
                    pt_5(ps) = pt(1);
                end
        end
 
        M = ps(1);
        lambda = lrange(pl(1));
        T = Trange(pt(1));
        
        % Load the precomputed results for the learnt best parameters
        load(['./savedir/' type 'solpairacc_' num2str(lambda) '.mat']);
        load(['./savedir/' type 'seg_sol_iou_en_' num2str(lambda) '.mat']);
        
        ntest1 = length(testIndices);
        sol_en1 = sol_en(:,testIndices);
        solpairacc1 = solpairacc(testIndices);
        sol_iou1 = sol_iou(:, testIndices);
        if(ii == 1) %Find oracle only for case 1
            % find oracle accuracies
            fprintf('Oracle: ');
            for ps=1:nummodes
                fprintf('%d...',ps);
                [sacc sind] = max(sol_iou1(1:ps,:),[],1);
                oracle_iou(ps) = mean(sacc(1,:));
            end
            fprintf('\n',ps);
        end
        
        fprintf('MBR: ');
        for ps = 1:nummodes
            fprintf('%d...',ps);
            
            if( ii == 5 )
                lambda = lrange(pl_5(ps));
                T = Trange(pt_5(ps));
                load(['./savedir/' type 'solpairacc_' num2str(lambda) '.mat']);
                load(['./savedir/' type 'seg_sol_iou_en_' num2str(lambda) '.mat']);
                sol_en1 = sol_en(:,testIndices);
                solpairacc1 = solpairacc(testIndices);
                sol_iou1 = sol_iou(:, testIndices);
            end
                
            for pf=1:ntest1
                prob = exp(-sol_en1(1:ps,pf)/T);
                prob = prob/sum(prob);
                
                % because we have accuracy not loss
                [mbr(pf) mbrind(pf)] = max(solpairacc1{pf}(1:ps,1:ps)*prob);
            end
            test_mbr_iou(ps) = mean(sol_iou1(sub2ind([nummodes ntest1], mbrind, 1:ntest1)));
            
        end
        fprintf('\n');
        case_struct.lambda = lambda;
        case_struct.T = T;
        case_struct.M = M;
        case_struct.test_mbr_iou = test_mbr_iou;
        
        switch ii
            case 1
                %lambda_m*, T_m*, m*
                case1 = case_struct;
            case 2
                %lambda_m*, T_inf, m*
                case2 = case_struct;
            case 3
                %lambda_M, T_inf, M
                case3 = case_struct;
            case 4
                %lambda_M, T_M, M
                case4 = case_struct;
            case 5
                % validate over 2 parameters for each m
                %lambda_m, T_m, m
                case5 = case_struct;
        end
        
    end
    save(test_set_results_fn, 'oracle_iou', 'case1', 'case2', 'case3', 'case4', 'case5');
else
    load(test_set_results_fn);
end

%% Plot the graphs
lw = 4;
fsize = 30;

if(strcmp(type, 'perturb'))
    type = 'Perturb&MAP';
end
if(strcmp(type, 'divMbest_boundary_'))
    type = 'DivMBest';
end

figure, hold on;
plot(100*oracle_iou,'k--s', 'linewidth',lw);
legendinfo{1} = 'Oracle';
plot(100*repmat(oracle_iou(1),1,nummodes),'r-.', 'linewidth',lw);
legendinfo{2} = 'MAP';
count = 2;
for ii = 3:5  % Do not plot the three parameter validation curve
    switch ii
%         If you want to plot the three parameter validation curve,
%         uncomment the following lines, loop through from 1:5 (above), and
%         change the indexing of the variable `legendinfo' (after the
%         switch case), from (ii-1) to ii.
%         
%         case 1
%             %lambda_m*, T_m*, m*
%             test_mbr_iou = case1.test_mbr_iou;
%             [m,ind] = max(test_mbr_iou);
%             c = 'b';            
%             leg = ['EMBR-' type ' (\lambda_{m*}=' num2str(case1.lambda) ', T_{m*}=' num2str(case1.T) ', m* = ' num2str(ind) ')'];
        case 2
            %lambda_m*, T_inf, m*
            test_mbr_iou = case2.test_mbr_iou;
            [m,ind] = max(test_mbr_iou);
            c = [0.17, 0.51, 0.34];
            leg = ['EMBR-' type ' (\lambda_{m*}, T=\infty, m*)'];
        case 3
            %lambda_M, T_inf, M
            test_mbr_iou = case3.test_mbr_iou;
            [m,ind] = max(test_mbr_iou);
            c = 'm';
            leg = ['EMBR-' type ' (\lambda_{M}, T=\infty, M)'];
        case 4
            %lambda_M, T_M, M
            test_mbr_iou = case4.test_mbr_iou;
            [m,ind] = max(test_mbr_iou);
            c = 'b';
            leg = ['EMBR-' type ' (\lambda_{M}, T_{M}, M=50)'];
        case 5
            % validate over 2 parameters for each m
            %lambda_m, T_m, m
            
            test_mbr_iou = case5.test_mbr_iou;
            [m,ind] = max(test_mbr_iou);
            c = [0.68, 0.47, 0];
            leg = ['EMBR-' type ' (\lambda_{m}, T_{m})'];            
    end
    plot(100*test_mbr_iou, '-s', 'Color', c, 'linewidth',lw);
    count = count+1;
    legendinfo{count} = leg;
end

% Dummy plots
plot(repmat(61.25,1,nummodes),'w', 'linewidth',lw);
str_iou = num2str((case1.test_mbr_iou(case1.M))*100);
str_iou = str_iou(1:5);
leg = ['EMBR-' type ' (\lambda_{m*}, T_{m*}, m*) = ' str_iou '%'];
count = count+1;
legendinfo{count} = leg;

plot(repmat(61.25,1,nummodes),'w', 'linewidth',lw);
str_iou = num2str((case2.test_mbr_iou(case2.M))*100);
str_iou = str_iou(1:5);
leg = ['EMBR-' type ' (\lambda_{m*}, T_{\infty}, m*) = ' str_iou '%'];
count = count+1;
legendinfo{count} = leg;

xlabel('# Solutions M', 'FontSize', fsize);
ylabel('Intersection-Over-Union (%)', 'FontSize', fsize);
ylim([61 78]);
legend(legendinfo);
set(gcf, 'Color', 'w');
set(gca,'fontsize',fsize,'linewidth',lw);
