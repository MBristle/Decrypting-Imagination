%% read imges in folder and process
% imges are read from the imgPath, 
% resized to the hight of the inteded size on the y axis,
% croped on the x axis to the inteded size of the x axis (by taking the
% centeral part of the image)
% converting the image to gray,
% setting the mean luminance of the image to 50% 
% images are saved to the specified output folder

numOfImg=90;
intendedSize=[1024,1280];
wantedMean=0.5;
imgPath='img/stim_org/';
outPath='img/stim/';
imgFiles=dir([imgPath,'*.jpg']);
assert(length(imgFiles)==numOfImg,'number of images does not match or an image is not ending with *.jpg')

for i= 1:numOfImg
    img=imread([imgPath,imgFiles(i).name]);
    [~, filename, extension] = fileparts([imgPath,imgFiles(i).name]);
    img = im2double(img);
    imgSize=size(img);
    scale=intendedSize(1)/imgSize(1);
    img=imresize(img,scale);
    img=imcrop(img,[size(img,2)/2-0.5*1280,0,1280,1024]); 
    img=rgb2gray(img);
    

    datMean = mean(img(:));
    imD = img+ wantedMean - datMean;
    imshow(imD)
    imOut=im2uint8(imD);
    imwrite(imOut,[outPath,filename,extension],'jpg')
end

