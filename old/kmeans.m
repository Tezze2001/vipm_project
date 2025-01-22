% imds = imageDatastore('./dataset/train_set');

function [X, y] = read_training_labelled_dataset(file_paths)
    training_labels = readtable(file_paths, Delimiter=',');
    base_path = './dataset/train_set/';
    X = [];
    y = [];

    net = alexnet;
    inputSize = [227 227];

    for i=1:height(training_labels)
        %features = neural_features(im2double(imread([base_path cell2mat(training_labels{i,1})])), net, inputSize);
        features = hand_crafted_features(im2double(imread([base_path cell2mat(training_labels{i,1})])));
        X = [X; features];
        y = [y; training_labels{i, 2}];
    end
end

function features = mean_std_RGB_features(img)
    means = mean(reshape(img, [], 3));
    stds = std(reshape(img, [], 3));
    features = [means stds];
end

function features = hand_crafted_features(img)
    hsvImage = rgb2hsv(img);
    
    % features on hsv rappresentations
    means = mean(reshape(hsvImage, [], 3));
    stds = std(reshape(hsvImage, [], 3));
    h_perc = prctile(reshape(hsvImage(:, :, 1), [], 1), [0, 25, 50, 76, 100]);
    s_perc = prctile(reshape(hsvImage(:, :, 2), [], 1), [0, 25, 50, 76, 100]);
    v_perc = prctile(reshape(hsvImage(:, :, 2), [], 1), [0, 25, 50, 76, 100]);
    hvs_features = [means stds h_perc s_perc v_perc];
    
    grayImage = rgb2gray(img); % Converti in scala di grigi
    glcm = graycomatrix(grayImage); % Matrice di co-occorrenza
    stats_comatrix = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});


    
    Gx = imfilter(double(grayImage), fspecial('sobel')'); 
    Gy = imfilter(double(grayImage), fspecial('sobel')); 
    gradientMagnitude = sqrt(Gx.^2 + Gy.^2);

    metrics_grad = [mean(gradientMagnitude(:)) std(gradientMagnitude(:))];

    features=  [hvs_features stats_comatrix.Contrast stats_comatrix.Correlation stats_comatrix.Energy stats_comatrix.Homogeneity metrics_grad];
end

function features = neural_features(img, net, input_size)
    img = imresize(img, input_size);

    features = activations(net, img, 'relu5', 'OutputAs', 'rows'); % alexnet from relu7 layer
end

function [accuracy] = kNN(X_train, y_train, X_val, y_val, train_dim, K)
    % centroids = zeros(251, size(X_train, 2));
    % for i=0:250 
    %     batch = X_train(i*train_dim + 1:i*train_dim+train_dim, :);
    %     centroids(i+1,:) = mean(batch,1);
    % end
    % knnModel = fitcknn(centroids, 0:250,'NumNeighbors', K);

    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', K);


    predictedLabels = predict(knnModel, X_val);

       
    correctPredictions = sum(y_val == predictedLabels);
    totalSamples = length(y_val);

    accuracy = correctPredictions / totalSamples;
end 

%%

[X, y] = read_training_labelled_dataset('./dataset/train_small.csv');


%%
k = 10; % Number of folds
cv = cvpartition(y, 'KFold', k, "Stratify", true); % Create k-fold partition 90-10

% Initialize storage for accuracy results
accuracy = zeros(k, 1);

for i = 1:k
    % Get training and test indices
    trainIdx = cv.training(i);
    validationIdx = cv.test(i);
    
    % Split the data into training and test sets
    X_train = X(trainIdx, :);
    y_train = y(trainIdx);
    X_val = X(validationIdx, :);
    y_val = y(validationIdx);
    
    [X_train, mu, sigma] = zscore(X_train);
    X_val = (X_val - mu) ./ sigma;

    accuracy(i) = kNN(X_train, y_train, X_val, y_val, 18, 1);
    
end

disp(['Mean accuracy: ' num2str(mean(accuracy))])
disp(['STD accuracy: ' num2str(std(accuracy))])