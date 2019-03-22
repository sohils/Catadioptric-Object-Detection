boxImage = imread('rover.JPG');
% figure;
% imshow(boxImage);
% title('Image of a Box');
boxImage = rgb2gray(boxImage);
boxPoints = detectSURFFeatures(boxImage);

for i = 8712
    img_string = strcat("../pixel-finder/rover_images/IMG_",string(i),".JPG");
    sceneImage = imread(img_string.char);
    % figure;
    % imshow(sceneImage);
    % title('Image of a Cluttered Scene');

    sceneImage = rgb2gray(sceneImage);

    scenePoints = detectSURFFeatures(sceneImage);

    [boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
    [sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

    boxPairs = matchFeatures(boxFeatures, sceneFeatures);

    matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
    matchedScenePoints = scenePoints(boxPairs(:, 2), :);
    figure;
    showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
        matchedScenePoints, 'montage');
    title('Putatively Matched Points (Including Outliers)');


%     [tform, inlierBoxPoints, inlierScenePoints] = ...
%         estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');

    disp(matchedBoxPoints.Count);
end
% figure;
% showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
%     inlierScenePoints, 'montage');
% title('Matched Points (Inliers Only)');