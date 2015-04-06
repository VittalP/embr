function all_mbr = embr_prediction_wrapper(T, nummodes, boxes_mmodes, det_mmodes, solpairacc, test, ALPHA_TRIM)

if(~exist('ALPHA_TRIM', 'var'))
ALPHA_TRIM = 0;
end

numTestExamples = length(det_mmodes);
for ps = 1:nummodes
    fprintf('%d...',ps);
    
    [mbr, mbrind, mbr_boxes] = embr_prediction(boxes_mmodes, det_mmodes,solpairacc,numTestExamples,ps, T, ALPHA_TRIM);
    det_mbr = PARSE_transback(mbr_boxes);
    all_det_mbr(ps).det_mbr = det_mbr;
    all_det_mbr(ps).mbr = mbr;
    all_det_mbr(ps).mbrind = mbrind;
    all_det_mbr(ps).mbr_boxes = mbr_boxes;

    det_mbr = all_det_mbr(ps).det_mbr;
    mbrind = all_det_mbr(ps).mbrind;
    mbr = all_det_mbr(ps).mbr;
    all_mbr(ps) = PARSE_eval_apk(det_mbr, test, 0.1, 0);
end
