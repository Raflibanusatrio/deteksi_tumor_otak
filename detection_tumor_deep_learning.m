function tumor_detection_gui()
    % Create figure and UI elements
    fig = uifigure('Name', 'Tumor Detection GUI');
    img_axes = uiaxes(fig);
    img_axes.Position = [50, 250, 400, 400];
    browse_button = uibutton(fig, 'Text', 'Browse', 'Position', [250, 200, 100, 30], 'ButtonPushedFcn', @(src, event) browseButtonPushed(src));
    detect_button = uibutton(fig, 'Text', 'Detect', 'Position', [150, 150, 100, 30], 'ButtonPushedFcn', @(src, event) detectButtonPushed(src));
    classify_button = uibutton(fig, 'Text', 'Classify', 'Position', [350, 150, 100, 30], 'ButtonPushedFcn', @(src, event) classifyButtonPushed(src));
    result_label = uilabel(fig, 'Text', '', 'Position', [200, 100, 200, 30]);

    % Load pre-trained CIBR model and tumor classification model
    load('catt.mat, 50');
    load('catt.mat, 100');
    
    % Initialize variables
    input_image = [];
    segmentation_mask = [];
    tumor_region = [];
    classification_result = '';

    % Callback function for Browse button
    function browseButtonPushed(src)
        [filename, filepath] = uigetfile({'*.jpg', '*.png'}, 'Select Image');
        if filename
            input_image = imread(fullfile(filepath, filename));
            imshow(input_image, 'Parent', img_axes);
            result_label.Text = '';
        end
    end

    % Callback function for Detect button
    function detectButtonPushed(src)
        if isempty(input_image)
            errordlg('Please select an image first.', 'Error');
            return;
        end
        
        % Preprocess input image
        preprocessed_image = imresize(input_image, [256, 256]);
        preprocessed_image = im2double(preprocessed_image);

        % Perform tumor segmentation
        segmentation_mask = semanticseg(preprocessed_image, cibr_model);
        
        % Display segmented tumor region
        tumor_region = input_image;
        tumor_region(segmentation_mask ~= 1) = 0;
        imshow(tumor_region, 'Parent', img_axes);
    end

    % Callback function for Classify button
    function classifyButtonPushed(src)
        if isempty(segmentation_mask)
            errordlg('Please perform tumor detection first.', 'Error');
            return;
        end
        
        % Perform tumor classification using a pre-trained CNN model
        tumor_region_resized = imresize(tumor_region, [227, 227]); % Resize to match the input size of the classification network
        tumor_region_resized = im2double(tumor_region_resized);
        classification_result = classify(tumor_classification_model, tumor_region_resized);
        
        % Display classification result
        result_label.Text = ['Classification Result: ' char(classification_result)];
    end
end
