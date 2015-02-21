function [pck miou] = eval_pck(ca, gt, thresh)

% function [pck miou] = eval_pck(ca, gt, thresh)
%
% DB edit: This function now also
% -- computes a tight bounding box around keypoints
% -- computes IOU of predicted bounding box with gt bounding box
% -- computes mean of IOU scores
%
% Dhruv Batra (dbatra -at- vt.edu)
% Edited: 05/24/2013


if nargin < 3
  thresh = 0.1;
end

assert(numel(ca) == numel(gt));

% Compute the scale of the ground truths
for n = 1:length(gt)
  gt(n).scale = max(max(gt(n).point, [], 1) - min(gt(n).point, [], 1) + 1, [], 2);
  gt(n).scale = squeeze(gt(n).scale);
  
  % DB (Note this code assumes there is only 1 gt bbox in the image)
  if nargout > 1
    x = gt(n).point(:,1);
    y = gt(n).point(:,2);
    gtbox = [min(x) min(y) max(x) max(y)];
  end
end

for n = 1:length(gt)
  dist = sqrt(sum((ca(n).point-gt(n).point).^2,2));
  tp(:,n) = dist <= thresh * gt(n).scale;
  
  % DB (Note this code assumes there is only 1 gt bbox in the image)
  if nargout > 1
      iou(n) = overlap(ca.point,gtbox);
  end

end

pck = mean(tp,2)';

% DB
miou = mean(iou);

