% Load the trained model
load('brain_tumor_detection_model.mat', 'net');

% Specify the test data directory
testDataDir = 'C:\Users\Ahmad\Desktop\Signals Project 2\Dataset\Testing2';

% Create an ImageDatastore for the test data
imdsTestNew = imageDatastore(testDataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize and convert images to grayscale within the ImageDatastore
imdsTestNew.ReadFcn = @(filename)imresize(im2gray(imread(filename)), targetSize(1:2));

% Classify the test images using the trained model
YPredTest = classify(net, imdsTestNew);
YTestNew = imdsTestNew.Labels;

% Find indices of correct predictions
correctIndices = find(YPredTest == YTestNew);

% Display 5 correct results
numCorrectToShow = min(5, numel(correctIndices));
figure;

for i = 1:numCorrectToShow
    subplot(1, numCorrectToShow, i);
    idx = correctIndices(i);
    img = readimage(imdsTestNew, idx);
    imshow(img);
    title(['Predicted: ' char(YPredTest(idx)) ', Actual: ' char(YTestNew(idx))]);
end
