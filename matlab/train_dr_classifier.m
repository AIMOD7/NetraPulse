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
IMG_SIZE    = 512;
NUM_CLASSES = 5;       % 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative
BATCH_SIZE  = 16;
MAX_EPOCHS  = 30;
INIT_LR     = 1e-4;

% Paths (relative to repo root — adjust if running from a different CWD)
APTOS_IMAGES_DIR = '../data/aptos/train_images';
APTOS_LABELS_CSV = '../data/aptos/train.csv';
IDRID_IMAGES_DIR = '../data/idrid/B. Disease Grading/1. Original Images/a. Training Set';
ONNX_OUTPUT_PATH = '../ml/deploy/models/netrapulse_resnet50.onnx';

%% =========================================================================
%  Data Loading — REAL DATA PATH
% =========================================================================
% Load APTOS 2019 labels
aptos_labels = readtable(APTOS_LABELS_CSV);
% Columns: id_code, diagnosis (0-4)

% Build imageDatastore from APTOS images
aptos_files = fullfile(APTOS_IMAGES_DIR, ...
    strcat(aptos_labels.id_code, '.png'));
aptos_grades = categorical(aptos_labels.diagnosis);

aptos_ds = imageDatastore(aptos_files, ...
    'Labels', aptos_grades, ...
    'ReadFcn', @(f) preprocess_for_training(f, IMG_SIZE));

% ---- OPTIONAL: Merge IDRiD -----------------------------------------------
% Uncomment the block below after downloading IDRiD.
%
% idrid_files  = dir(fullfile(IDRID_IMAGES_DIR, '*.jpg'));
% idrid_labels_tbl = readtable('../data/idrid/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv');
% idrid_grades = categorical(idrid_labels_tbl.Retinopathy_grade);
% idrid_ds = imageDatastore({idrid_files.name}, 'Labels', idrid_grades, ...
%     'FileEncoding', 'auto', ...
%     'ReadFcn', @(f) preprocess_for_training(f, IMG_SIZE));
% combined_ds = combine(aptos_ds, idrid_ds);
% ds = combined_ds;
%
% For now, use APTOS only:
ds = aptos_ds;

%% =========================================================================
%  Train / Validation Split (80/20)
% =========================================================================
[trainDs, valDs] = splitEachLabel(ds, 0.8, 'randomize');
fprintf('Training samples:   %d\n', numel(trainDs.Files));
fprintf('Validation samples: %d\n', numel(valDs.Files));

%% =========================================================================
%  Model — ResNet50 with Transfer Learning
% =========================================================================
net_base = resnet50;
lgraph   = layerGraph(net_base);

% Replace the final FC layer for 5-class output
new_fc = fullyConnectedLayer(NUM_CLASSES, 'Name', 'fc_dr', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
new_softmax = softmaxLayer('Name', 'softmax_dr');
new_cls     = classificationLayer('Name', 'output_dr');

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
    % Replicates ml/data/preprocessing.py logic in MATLAB
    % Crop → Resize → Circular mask → Ben Graham normalization
    img = imread(filepath);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, [img_size, img_size]);

    % Circular mask
    [h, w, ~] = size(img);
    [X, Y] = meshgrid(1:w, 1:h);
    cx = w/2; cy = h/2; r = min(h, w)/2;
    mask = ((X-cx).^2 + (Y-cy).^2) <= r^2;
    for c = 1:3
        ch = double(img(:,:,c));
        ch(~mask) = 0;
        img(:,:,c) = uint8(ch);
    end

    % Ben Graham local contrast normalization
    sigma = 10;
    img_d = double(img);
    blurred = imgaussfilt(img_d, sigma);
    img_d = img_d * 4 + blurred * (-4) + 128;
    img_d = max(0, min(255, img_d));
    img = uint8(img_d);
end
