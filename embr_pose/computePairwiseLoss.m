function [mpck, pck, iou] = computePairwiseLoss(det_mmodes, nummodes, pim)

det1 = []; det2 = [];
mpck = []; iou = [];
for ps1 = 1:nummodes
    det1.point = det_mmodes(pim).point(:,:,ps1);
    det1.score = det_mmodes(pim).score(ps1);
    for ps2 = 1:nummodes
        det2.point = det_mmodes(pim).point(:,:,ps2);
        det2.score = det_mmodes(pim).score(ps2);
        
        [mpck(ps1,ps2), pck, iou(ps1,ps2)] = PARSE_eval_pck(det1, det2, 0.1, 0);
        
    end
end
