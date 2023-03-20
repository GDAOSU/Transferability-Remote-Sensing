% extraction of blob shaped structures

function y = tophat_reconstruction_disk_shape(img,radius)
% the first things is to fill the NULL values.
% also this needs to be segmented.
minv = min(img(:));
indnan = find(isnan(img));
img(indnan) = minv;

se = strel('disk',radius,8);

imer = imerode(img,se);

imgrecon = imreconstruct(imer,img);

y = img - imgrecon;

y(indnan) = NaN;

end