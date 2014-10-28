function [mbr, mbrind, mbr_boxes] = mbr_prediction(boxes_mmodes, det_mmodes, solpairacc, numTestExamples, ps, T)

if(~exist('T', 'var'))
    T = 0.1;
end

mbr_boxes = [];
for pim=1:numTestExamples
    prob = exp(det_mmodes(pim).score(1:ps)'/T);
    prob = prob/sum(prob);
    
    % maximum bayes risk because we have accuracy not loss
    [mbr(pim), mbrind(pim)] = max(solpairacc{pim}(1:ps,1:ps)*prob);
    [mbr_sort, mbrind_sort] = sort(solpairacc{pim}(1:ps,1:ps)*prob);
    mbr_boxes{pim} = boxes_mmodes{pim}(mbrind_sort,:);
    mbr_boxes{pim}(:,end) = mbr_sort;
    
    mbr_boxes{pim} = nms(mbr_boxes{pim},0.3);
end