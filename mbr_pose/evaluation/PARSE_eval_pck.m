function [meanpck pck miou] = PARSE_eval_pck(det_gtbox, test, thres, verbose)

% function [meanpck pck] = PARSE_eval_pck(det_gtbox, test, thres)
% evaluation 2: percentage of correct keypoints
% You will need to write your own PCK evaluation code for your data structure

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = 1;
end

[pck miou] = eval_pck(det_gtbox, test, thres);
% Average left with right and neck with top head
pck = (pck + pck([6 5 4 3 2 1 12 11 10 9 8 7 14 13]))/2;
% Change the order to: Head & Shoulder & Elbow & Wrist & Hip & Knee & Ankle
pck = pck([14 9 8 7 3 2 1]);

meanpck = mean(pck);

if verbose
    fprintf('mean PCK = %.1f\n',meanpck*100); 
    fprintf('Keypoints & Head & Shou & Elbo & Wris & Hip  & Knee & Ankle\n');
    fprintf('PCK       '); fprintf('& %.1f ',pck*100); fprintf('\n');
end
