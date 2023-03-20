clear all
close all
dsmname = 'D:/Reasearch/transferability analysis/Data/AHS/Argentina/DSM.tif';
radius = 50;
dsm = imread(dsmname);
y = tophat_reconstruction_disk_shape(dsm,radius);
t = saveAsTiffSingle(y, 'AGL.tif');
