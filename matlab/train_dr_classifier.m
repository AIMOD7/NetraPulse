% matlab/train_dr_classifier.m
%
% NetraPulse — Diabetic Retinopathy Classifier Training
% Trains a ResNet50 on APTOS 2019 + IDRiD fundus images.
%
% NOTE: Synthetic data generation (generate_synthetic_fundus.m) is now
%       used ONLY as a fallback / unit-test fixture. The primary data
%       path reads from real datasets below.
%
% After training, the model is exported to ONNX for the FastAPI service.
%
% Usage:
%   Run this script in MATLAB with the Deep Learning Toolbox Converter
%   for ONNX installed (Add-On Explorer → search "ONNX Model Format").

%% =========================================================================
%  Configuration
% =========================================================================
IMG_SIZE    = 224;
NUM_CLASSES = 5;       % 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
BATCH_SIZE  = 16;
MAX_EPOCHS  = 2;       % Smoke test limit
INIT_LR     = 1e-4;

% ---------------------------------------------------------------------------
% DATA PATHS  (images are already cropped/resized to 1024x1024 — no need to
% run generate_synthetic_fundus.m or the APTOS download script)
% ---------------------------------------------------------------------------
IMAGES_DIR   = '../resized_train_cropped';   % 35,108 JPEG files: {id}_{left|right}.jpeg
LABELS_CSV   = '../trainLabels.csv';         % Columns: image, level (0-4)
ONNX_OUTPUT_PATH = '../ml/deploy/models/netrapulse_resnet50.onnx';

%% =========================================================================
%  Data Loading — resized_train_cropped (pre-cropped 1024x1024 images)
% =========================================================================
% Load labels from trainLabels.csv
% Columns: image (e.g. "10_left"), level (0-4)
train_labels = readtable(LABELS_CSV);
% Rename for clarity
train_labels.Properties.VariableNames = {'image', 'level'};

fprintf('Loaded %d label entries from %s\n', height(train_labels), LABELS_CSV);

% Build full paths: {id}_{left|right}.jpeg
image_files = fullfile(IMAGES_DIR, strcat(train_labels.image, '.jpeg'));

% Remove any files that don't exist on disk
exists_mask = cellfun(@(f) isfile(f), image_files);
image_files   = image_files(exists_mask);
grades        = categorical(train_labels.level(exists_mask));

fprintf('Found %d/%d images on disk.\n', sum(exists_mask), numel(exists_mask));

% Build imageDatastore
ds = imageDatastore(image_files, ...
    'Labels', grades, ...
    'ReadFcn', @(f) preprocess_for_training(f, IMG_SIZE));

%% =========================================================================
%  Train / Validation Split (80/20) - SMOKE TEST LIMIT
% =========================================================================
% Shuffle and extract only 1000 images to test the pipeline quickly
ds = shuffle(ds);
ds_small = subset(ds, 1:min(1000, numel(ds.Files)));

[trainDs, valDs] = splitEachLabel(ds_small, 0.8, 'randomize');
fprintf('Training samples:   %d\n', numel(trainDs.Files));
fprintf('Validation samples: %d\n', numel(valDs.Files));

%% =========================================================================
%  Model — ResNet50 with Transfer Learning & Class Weighting
% =========================================================================
net_base = resnet50;
lgraph   = layerGraph(net_base);

% Calculate inverse-frequency class weights to handle dataset imbalance
labelCounts = countEachLabel(trainDs);
classWeights = sum(labelCounts.Count) ./ (height(labelCounts) * labelCounts.Count);

% Replace the final FC layer for 5-class output
new_fc = fullyConnectedLayer(NUM_CLASSES, 'Name', 'fc_dr', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
new_softmax = softmaxLayer('Name', 'softmax_dr');

% Apply the calculated weights to the new classification layer
new_cls = classificationLayer('Name', 'output_dr', ...
    'Classes', labelCounts.Label, ...
    'ClassWeights', classWeights);

lgraph = replaceLayer(lgraph, 'fc1000',       new_fc);
lgraph = replaceLayer(lgraph, 'fc1000_softmax', new_softmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', new_cls);

%% =========================================================================
%  Training Options
% =========================================================================
options = trainingOptions('adam', ...
    'InitialLearnRate',   INIT_LR, ...
    'MaxEpochs',          MAX_EPOCHS, ...
    'MiniBatchSize',      BATCH_SIZE, ...
    'Shuffle',            'every-epoch', ...
    'ValidationData',     valDs, ...
    'ValidationFrequency', 10, ...
    'Plots',              'training-progress', ...
    'Verbose',            true);

%% =========================================================================
%  Train
% =========================================================================
fprintf('\nStarting training...\n');
[net, info] = trainNetwork(trainDs, lgraph, options);
fprintf('Training complete. Final val accuracy: %.2f%%\n', ...
    max(info.ValidationAccuracy));

%% =========================================================================
%  ONNX Export
% =========================================================================
fprintf('\nExporting to ONNX: %s\n', ONNX_OUTPUT_PATH);
exportONNXNetwork(net, ONNX_OUTPUT_PATH);
fprintf('Export complete.\n');
fprintf('Verify with: python ml/scripts/verify_onnx.py\n');

%% =========================================================================
%  Local Preprocessing Helper
% =========================================================================
function img = preprocess_for_training(filepath, img_size)
    % Preprocessing for resized_train_cropped images (already 1024x1024,
    % already cropped — skip content-crop, go straight to resize+mask+normalize).
    img = imread(filepath);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    % Resize 1024x1024 → img_size x img_size
    img = imresize(img, [img_size, img_size]);

    % Circular mask (drop corners outside the retinal disc)
    [h, w, ~] = size(img);
    [X, Y] = meshgrid(1:w, 1:h);
    cx = w/2; cy = h/2; r = min(h, w)/2;
    mask = ((X-cx).^2 + (Y-cy).^2) <= r^2;
    img_d = double(img);
    for c = 1:3
        ch = img_d(:,:,c);
        ch(~mask) = 0;
        img_d(:,:,c) = ch;
    end

    % Ben Graham local-contrast normalization
    sigma = 10;
    blurred = imgaussfilt(img_d, sigma);
    img_d   = img_d * 4 + blurred * (-4) + 128;
    img_d   = max(0, min(255, img_d));
    img = uint8(img_d);
end