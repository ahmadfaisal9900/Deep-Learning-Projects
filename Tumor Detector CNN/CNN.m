dataDir = 'C:\Users\Ahmad\Desktop\Signals Project 2\Dataset\Training2';

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Specify the target size
targetSize = [256 256 1]; % Change the third dimension to 1 for grayscale

% Resize and convert images to grayscale within the ImageDatastore
imds.ReadFcn = @(filename)imresize(im2gray(imread(filename)), targetSize(1:2));

% Split the dataset into training and testing sets (e.g., 80% for training, 20% for testing)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Display the size of the first image in the training set
sampleImage = read(imdsTrain);
disp(size(sampleImage));

layers = [
    imageInputLayer(targetSize)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4) % Four output classes
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Plots', 'training-progress', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30);

net = trainNetwork(imdsTrain, layers, options);

YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

save('brain_tumor_detection_model.mat', 'net');
