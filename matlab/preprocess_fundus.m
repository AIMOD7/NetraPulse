% matlab/preprocess_fundus.m
%
% NetraPulse — Fundus Image Preprocessing
%
% NOTE: Synthetic data generation (generate_synthetic_fundus.m) is now
%       used ONLY as a fallback / unit-test fixture. The primary data
%       path reads from real datasets (../data/aptos, ../data/idrid).
%
% This function mirrors ml/data/preprocessing.py exactly so that
% train-time and serve-time preprocessing are identical.

function img_out = preprocess_fundus(img_in, img_size, sigma)
    % preprocess_fundus  Circle-crop + local-contrast normalization.
    %
    %   img_out = preprocess_fundus(img_in)
    %   img_out = preprocess_fundus(img_in, img_size, sigma)
    %
    %   img_in  : uint8 RGB image
    %   img_size: output resolution (default 512)
    %   sigma   : Gaussian blur sigma for Ben Graham normalization (default 10)

    if nargin < 2, img_size = 512; end
    if nargin < 3, sigma    = 10;  end

    % Ensure 3-channel
    if size(img_in, 3) == 1
        img_in = cat(3, img_in, img_in, img_in);
    end

    % 1. Crop to non-black content
    img_in = crop_to_content(img_in, 7);

    % 2. Resize
    img_in = imresize(img_in, [img_size, img_size]);

    % 3. Circular mask
    [h, w, ~] = size(img_in);
    [X, Y] = meshgrid(1:w, 1:h);
    cx = w/2; cy = h/2; r = min(h, w)/2;
    mask = ((X-cx).^2 + (Y-cy).^2) <= r^2;
    img_d = double(img_in);
    for c = 1:3
        ch = img_d(:,:,c);
        ch(~mask) = 0;
        img_d(:,:,c) = ch;
    end

    % 4. Ben Graham local-contrast normalization
    blurred = imgaussfilt(img_d, sigma);
    img_d   = img_d * 4 + blurred * (-4) + 128;
    img_d   = max(0, min(255, img_d));
    img_out = uint8(img_d);
end


function img = crop_to_content(img, tol)
    % Remove dark borders based on grayscale threshold.
    gray = rgb2gray(img);
    mask = gray > tol;
    if ~any(mask(:))
        return;
    end
    rows = any(mask, 2);
    cols = any(mask, 1);
    r1 = find(rows, 1, 'first');
    r2 = find(rows, 1, 'last');
    c1 = find(cols, 1, 'first');
    c2 = find(cols, 1, 'last');
    img = img(r1:r2, c1:c2, :);
end
