function t = saveAsTiffSingle(X, filepath)
    im = single(X);
    t = Tiff(filepath,'w');
    tagstruct.ImageLength = size(im,1); 
    tagstruct.ImageWidth = size(im,2);  

    tagstruct.Photometric = 1;
    tagstruct.BitsPerSample = 32;

    tagstruct.SamplesPerPixel = 1;
%     tagstruct.RowsPerStrip = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;

    tagstruct.Software = 'MATLAB';

    tagstruct.SampleFormat = 3;
    t.setTag(tagstruct)

    t.write(im);

    t.close
end