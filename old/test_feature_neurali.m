net = alexnet;
analyzeNetwork(net)

sz = net.Layers(1).InputSize;

%% Estrazione delle immagini 
% Percorsi dei file 
csv_tr = './dataset/train_small.csv';
csv_te = './dataset/val_info.csv'; 
indir_tr = './dataset/train_set';
indir_te = './dataset/val_set';
indir_te_degraded = './dataset/val_set_degraded';

% Legge i dati dal file CSV (Training)
data_tr = readtable(csv_tr, 'Format', '%s%d', 'Delimiter', ',');
imageNames_tr_small = data_tr{:, 1};  % Estrae i nomi delle immagini
labels_tr_small = data_tr{:, 2};      % Estrae le etichette

% Legge i dati dal file CSV (Validation)
data_te = readtable(csv_te, 'Format', '%s%d', 'Delimiter', ',');
imageNames_te = data_te{:, 1};  % Estrae i nomi delle immagini
labels_te = data_te{:, 2};      % Estrae le etichette

% Ottiene tutti i file immagine nella cartella di validazione
all_images_te = dir(fullfile(indir_te, '*.jpg'));
%% Layer 
layer = 'relu5'; 

%% Inizializzazione delle variabili
num_tr = length(imageNames_tr_small);
num_te = length(imageNames_te);
feat_size = size(activations(net, rand(sz), layer, 'OutputAs', 'rows'), 2); % Dimensioni feature

feat_tr = zeros(num_tr, feat_size); % Preallocazione
labels_tr = zeros(num_tr, 1);
feat_te = zeros(num_te, feat_size); % Preallocazione

tic
disp('Estrazione delle feature...');

%% Training set
for i = 1:num_tr
    disp(['[Training] Immagine: ', num2str(i)]);
    im_path = fullfile(indir_tr, imageNames_tr_small{i}); 
    im = imread(im_path); 

    % Converti in RGB se in scala di grigi
    if size(im, 3) == 1
        im = repmat(im, [1, 1, 3]);
    end

    % Ridimensiona solo se necessario
    if ~isequal(size(im, 1:2), sz(1:2))
        im = imresize(im, sz(1:2)); 
    end

    % Estrai le feature
    feat_tr(i, :) = activations(net, im, layer, 'OutputAs', 'rows');
    labels_tr(i) = labels_tr_small(i);
end

%% Test set
for i = 1:num_te
    disp(['[Test] Immagine: ', num2str(i)]);
    im_path = fullfile(indir_te, imageNames_te{i}); 
    im = imread(im_path);

    % Converti in RGB se in scala di grigi
    if size(im, 3) == 1
        im = repmat(im, [1, 1, 3]);
    end

    % Ridimensiona solo se necessario
    if ~isequal(size(im, 1:2), sz(1:2))
        im = imresize(im, sz(1:2)); 
    end

    % Estrai le feature
    feat_te(i, :) = activations(net, im, layer, 'OutputAs', 'rows');
end
toc


%% visualizzazione dei filtri
feat=activations(net,im,layer);
figure(1), clf
for ii=1:25
    subplot(5,5,ii)
    imagesc(feat(:,:,ii)), colormap gray, drawnow
end

%% normalizzazione features
feat_tr = feat_tr./sqrt(sum(feat_tr.^2,2));
feat_te = feat_te./sqrt(sum(feat_te.^2,2));
save features_cibo_relu5.mat feat_tr labels_tr feat_te labels_te

%% classificazione con 1-NN
load features_cibo_relu5.mat
layer = 'relu5'; 
D=pdist2(feat_te,feat_tr);
[dummy,idx_pred_te]=min(D,[],2);
lab_pred_te = labels_tr(idx_pred_te);
acc = numel(find(lab_pred_te==labels_te))/numel(labels_te);

labels_te = double(labels_te);
lab_pred_te = double(lab_pred_te);

confusion_matrix = confusionmat(labels_te, lab_pred_te);
figure;
confusionchart(confusion_matrix);
title(sprintf('Accuratezza: %.2f%% - Layer: %s%', acc * 100, layer));