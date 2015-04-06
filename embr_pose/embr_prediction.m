function [mbr, mbrind, mbr_boxes] = embr_prediction(boxes_mmodes, det_mmodes, solpairacc, numTestExamples, ps, T, ALPHA_TRIM)

if(~exist('T', 'var'))
    T = 0.1;
end

if(~exist('ALPHA_TRIM', 'var'))
    ALPHA_TRIM = 0;
end

mbr_boxes = [];
for pim=1:numTestExamples
    prob_orig = exp(det_mmodes(pim).score(1:ps)'/T);
    
    if(ALPHA_TRIM)
    	% embr_score_vec = zeros(ps, 1);
    	% for pps = 1:ps
    	% 	prob = prob_orig;
    	% 	robust_idx = solpairacc{pim}(pps,1:ps) >= ALPHA_TRIM;
    	% 	prob = prob(robust_idx);
    	% 	prob = prob/sum(prob);

    	% 	embr_score_vec(pps) = solpairacc{pim}(pps, robust_idx) * prob;
    	% end

        temp = solpairacc{pim}(1:ps, 1:ps);
        outlier_idx = temp < ALPHA_TRIM;
        temp(outlier_idx) = ALPHA_TRIM;
    else
        temp = solpairacc{pim}(1:ps, 1:ps);
    end

    	
    	prob = prob_orig;
    	prob = prob/sum(prob);

	    % maximum bayes risk because we have accuracy not loss
	    % [mbr(pim), mbrind(pim)] = max(solpairacc{pim}(1:ps,1:ps)*prob);
	    % [mbr_sort, mbrind_sort] = sort(solpairacc{pim}(1:ps,1:ps)*prob);

        [mbr(pim), mbrind(pim)] = max(temp*prob);
        [mbr_sort, mbrind_sort] = sort(temp*prob);


    mbr_boxes{pim} = boxes_mmodes{pim}(mbrind_sort,:);
    mbr_boxes{pim}(:,end) = mbr_sort;
    
    mbr_boxes{pim} = nms(mbr_boxes{pim},0.3);
end
