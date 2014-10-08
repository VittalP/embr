function [meanapk apk] = PARSE_eval_apk(det, test, thres, verbose)

% function [meanapk apk] = PARSE_eval_apk(det, test, thres)
% evaluation 1: average precision of keypoints
% You will need to write your own APK evaluation code for your data structure

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = 1;
end

apk = eval_apk(det, test, thres);
% Average left with right and neck with top head
apk = (apk + apk([6 5 4 3 2 1 12 11 10 9 8 7 14 13]))/2;
% Change the order to: Head & Shoulder & Elbow & Wrist & Hip & Knee & Ankle
apk = apk([14 9 8 7 3 2 1]);
meanapk = mean(apk);

if verbose
    fprintf('mean APK = %.1f\n',meanapk*100);
    fprintf('Keypoints & Head & Shou & Elbo & Wris & Hip  & Knee & Ankle\n');
    fprintf('APK       '); fprintf('& %.1f ',apk*100); fprintf('\n');
end
