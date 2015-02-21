function iou = overlap(point,gtbox)

% function iou = overlap(point,gtbox)
%
% Function computes Intersection-Over-Union (IOU) between a set of
% detections represented by points -- <nkeypoints> (14) x 2 x <ndets>
% and a SINGLE gtbox -- 1 x 4 (xleft yleft xright y right). 
% Most of the code taken from bestoverlap.m
%
% Dhruv Batra (dbatra -at- vt.edu)
% Created: 05/24/2013

iou = 0;
if isempty(point) || isempty(gtbox)
  return;
end

assert(isvector(gtbox)==1);
assert(length(gtbox)==4);


x1 = gtbox(1); y1 = gtbox(2); x2 = gtbox(3); y2 = gtbox(4);
area_gt = (x2-x1+1).*(y2-y1+1);

% % ignore score if present in boxes
% b = boxes(:,1:floor(size(boxes, 2)/4)*4);
% % reshape to <nbox> x 4 x <nparts>
% b = reshape(b,size(b,1),4,size(b,2)/4);
% 
% % (x,y) center of parts
% bx = .5*b(:,1,:) + .5*b(:,3,:);
% by = .5*b(:,2,:) + .5*b(:,4,:);
% 
% % tightest bounding box around centers
% bx1 = min(bx,[],3);
% bx2 = max(bx,[],3);
% by1 = min(by,[],3);
% by2 = max(by,[],3);

% tightest bounding box around keypoints
x = point(:,1,:);
y = point(:,2,:);

bboxes = [min(x,[],1) min(y,[],1) max(x,[],1) max(y,[],1)];

% reshape so that first dim is <ndets> 
bboxes = permute(bboxes,[3 2 1]);

bx1 = bboxes(:,1);
bx2 = bboxes(:,3);
by1 = bboxes(:,2);
by2 = bboxes(:,4);

% intersection of boxes & gtbox
xx1 = max(x1,bx1);
yy1 = max(y1,by1);
xx2 = min(x2,bx2);
yy2 = min(y2,by2);

w = xx2-xx1+1; w(w<0) = 0;
h = yy2-yy1+1; h(h<0) = 0;
inter = w.*h;

% iou
area_boxes = (bx2-bx1+1).*(by2-by1+1);
union = area_boxes + area_gt - inter;

iou = inter ./ union;
