net = alexnet;
% net = alexnet('Weights','none');
sz = net.Layers(1).InputSize;

%% cut layers
layersTransfer = net.Layers(1:end-6);
% layersTransfer = freezeWeights(layersTransfer);
% analyzeNetwork(layersTransfer)

%% replace layers
numClasses = 251;
% layers = [layersTransfer
%     fullyConnectedLayer(numClasses);
%     softmaxLayer
%     classificationLayer];

layers = [layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',200,...
    'BiasLearnRateFactor',200);
    softmaxLayer
    classificationLayer];

% analyzeNetwork(layers)


%% Preparazione dati
% Percorsi dei file CSV e delle directory
trainCSV = './dataset/train_small.csv';
testCSV = './dataset/val_info.csv';
trainPath = './dataset/train_set';
testPath = './dataset/val_set';

% Legge i file CSV
dataTrain = readtable(trainCSV, 'Format', '%s%d', 'Delimiter', ',');
dataTest = readtable(testCSV, 'Format', '%s%d', 'Delimiter', ',');

% Nomi immagini e etichette per il training
imageNamesTrain = dataTrain{:, 1}; % Colonna con i nomi delle immagini
labelsTrain = categorical(dataTrain{:, 2}); % Colonna con le etichette (categorical)

% Nomi immagini e etichette per il test
imageNamesTest = dataTest{:, 1}; % Colonna con i nomi delle immagini
labelsTest = categorical(dataTest{:, 2}); % Colonna con le etichette (categorical)

% Creazione dei fileDatastore per il training
imdsTrain = imageDatastore(fullfile(trainPath, imageNamesTrain), ...
    'Labels', labelsTrain);

% Creazione del fileDatastore per il test
imdsTest = imageDatastore(fullfile(testPath, imageNamesTest), ...
    'Labels', labelsTest);

% Suddivisione del set di training in training e validazione
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain, 0.8, 'randomized'); % 80% training, 20% validation

% Visualizzazione delle distribuzioni delle etichette
disp('Distribuzione delle etichette nel training set:');
countEachLabel(imdsTrain)

disp('Distribuzione delle etichette nel validation set:');
countEachLabel(imdsValidation)

disp('Distribuzione delle etichette nel test set:');
countEachLabel(imdsTest)

%% Data Augmentation
pixelRange = [-5 5];
imageAugmenter = imageDataAugmenter(...
   'RandXReflection', true, ...
   'RandXTranslation', pixelRange, ...
   'RandYTranslation', pixelRange);

% Creazione dei datastore aumentati
augimdsTrain = augmentedImageDatastore(sz(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(sz(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsTest = augmentedImageDatastore(sz(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% configurazione training

% Nuove opzioni di training
options = trainingOptions('adam', ...
    'MiniBatchSize', 256, ... 
    'MaxEpochs', 4, ...      
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 2, ...
    'LearnRateDropFactor', 0.1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% training vero e proprio
netTransfer = trainNetwork(augimdsTrain,layers,options);

%% test
tic
[lab_pred_te,scores] = classify(netTransfer,augimdsTest);
toc

%% valutazione performance
acc = numel(find(lab_pred_te==imdsTest.Labels))/numel(lab_pred_te)