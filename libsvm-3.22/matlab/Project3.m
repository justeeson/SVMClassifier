% Name: Sebastin Justeeson
% Description: Classification using LIBSVM

% Read in the testing and training data
[test_label_vector, test_instance_matrix] = libsvmread('ncrna_s.test'); 
[train_label_vector, train_instance_matrix] = libsvmread('ncrna_s.train'); 

% Convert sparse matrices to full matrices
train_instance_matrix = full(train_instance_matrix);
test_instance_matrix = full(test_instance_matrix);
train_label_vector = full(train_label_vector);
test_label_vector = full(test_label_vector);

% Define all the models
model_4 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 0.0625');
model_3 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 0.125');
model_2 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 0.25');
model_1 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 0.5');
model0 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 1'); 
model1 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 2'); 
model2 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 4'); 
model3 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 8'); 
model4 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 16'); 
model5 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 32'); 
model6 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 64'); 
model7 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 128'); 
model8 = svmtrain(train_label_vector, train_instance_matrix, '-t 0 -c 256'); 

% Predict the accuracy of the model on the test set
[predict_label, accuracy1, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model_4);
[predict_label, accuracy2, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model_3);
[predict_label, accuracy3, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model_2);
[predict_label, accuracy4, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model_1);
[predict_label, accuracy5, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model0);
[predict_label, accuracy6, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model1);
[predict_label, accuracy7, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model2);
[predict_label, accuracy8, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model3);
[predict_label, accuracy9, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model4);
[predict_label, accuracy10, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model5);
[predict_label, accuracy11, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model6);
[predict_label, accuracy12, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model7);
[predict_label, accuracy13, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model8);

% Convert the accuracies from percentages to decimals and put into an array
x = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];
%y = [0.678, 0.678, 0.6775, 0.788, 0.941, 0.953, 0.9525, 0.953, 0.9525, 0.953, 0.9525, 0.9525, 0.9525];
z = [accuracy1(1)/100, accuracy2(1)/100, accuracy3(1)/100, accuracy4(1)/100, accuracy5(1)/100, accuracy6(1)/100, accuracy7(1)/100, accuracy8(1)/100, accuracy9(1)/100, accuracy10(1)/100, accuracy11(1)/100, accuracy12(1)/100, accuracy13(1)/100];

% Plot the array
semilogx(x,z);
title('Accuracy vs C value');
xlabel('C value');
ylabel('Accuracy');

% Set 50% of the training set as the cross validation set.
 s = RandStream('mlfg6331_64'); 
 y = datasample(s,1:2000,1000,'Replace',false);
 validation_instance_matrix = zeros(1000, 8);
 validation_label_vector = zeros(1000, 1);
 for i = 1:1000
     validation_instance_matrix(i, 1:8) = train_instance_matrix(y(i), 1:8);
     validation_label_vector(i, 1) = train_label_vector(y(i), 1);
 end
 
% Divide the cross validation set into 5 subsets
validation_instance_1 = validation_instance_matrix(1:200, 1:8);
validation_label_1 = validation_label_vector(1:200, 1);
validation_instance_2 = validation_instance_matrix(201:400, 1:8);
validation_label_2 = validation_label_vector(201:400, 1);
validation_instance_3 = validation_instance_matrix(401:600, 1:8);
validation_label_3 = validation_label_vector(401:600, 1);
validation_instance_4 = validation_instance_matrix(601:800, 1:8);
validation_label_4 = validation_label_vector(601:800, 1);
validation_instance_5 = validation_instance_matrix(801:1000, 1:8);
validation_label_5 = validation_label_vector(801:1000, 1);

% Define the C and ? sets
C = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];
ALPHA = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];
accuracy_matrix = zeros(13,13);

% 5-fold cross validation
for i = 1:13
    for j = 1:13
        alphaval = ALPHA(j);
        cval = C(i);
       
        options_string = '-c';
        options_string = strcat({options_string}, {' '}, {num2str(cval)});
        options_string = strcat(options_string, {' '}, '-g');
        options_string = strcat(options_string, {' '}, {num2str(alphaval)});
        
        options_string = char(options_string);
        
        RBFmodel_1 = svmtrain(validation_label_1, validation_instance_1, options_string);  
        RBFmodel_2 = svmtrain(validation_label_2, validation_instance_2, options_string);
        RBFmodel_3 = svmtrain(validation_label_3, validation_instance_3, options_string);
        RBFmodel_4 = svmtrain(validation_label_4, validation_instance_4, options_string);
        RBFmodel_5 = svmtrain(validation_label_5, validation_instance_5, options_string);
        
        % Apply the SVMs to each validation subset
        [predict_label, RBF_accuracy1, dec_values] = svmpredict(validation_label_1, validation_instance_1, RBFmodel_2);
        [predict_label, RBF_accuracy2, dec_values] = svmpredict(validation_label_1, validation_instance_1, RBFmodel_3);
        [predict_label, RBF_accuracy3, dec_values] = svmpredict(validation_label_1, validation_instance_1, RBFmodel_4);
        [predict_label, RBF_accuracy4, dec_values] = svmpredict(validation_label_1, validation_instance_1, RBFmodel_5);
        accuracy1 = (RBF_accuracy1(1) + RBF_accuracy2(1) + RBF_accuracy3(1) + RBF_accuracy4(1))/4;
        
        [predict_label, RBF_accuracy1, dec_values] = svmpredict(validation_label_2, validation_instance_2, RBFmodel_1);
        [predict_label, RBF_accuracy2, dec_values] = svmpredict(validation_label_2, validation_instance_2, RBFmodel_3);
        [predict_label, RBF_accuracy3, dec_values] = svmpredict(validation_label_2, validation_instance_2, RBFmodel_4);
        [predict_label, RBF_accuracy4, dec_values] = svmpredict(validation_label_2, validation_instance_2, RBFmodel_5);
        accuracy2 = (RBF_accuracy1(1) + RBF_accuracy2(1) + RBF_accuracy3(1) + RBF_accuracy4(1))/4;
        
        [predict_label, RBF_accuracy1, dec_values] = svmpredict(validation_label_3, validation_instance_3, RBFmodel_1);
        [predict_label, RBF_accuracy2, dec_values] = svmpredict(validation_label_3, validation_instance_3, RBFmodel_2);
        [predict_label, RBF_accuracy3, dec_values] = svmpredict(validation_label_3, validation_instance_3, RBFmodel_4);
        [predict_label, RBF_accuracy4, dec_values] = svmpredict(validation_label_3, validation_instance_3, RBFmodel_5);
         accuracy3 = (RBF_accuracy1(1) + RBF_accuracy2(1) + RBF_accuracy3(1) + RBF_accuracy4(1))/4;
         
        [predict_label, RBF_accuracy1, dec_values] = svmpredict(validation_label_4, validation_instance_4, RBFmodel_1);
        [predict_label, RBF_accuracy2, dec_values] = svmpredict(validation_label_4, validation_instance_4, RBFmodel_2);
        [predict_label, RBF_accuracy3, dec_values]= svmpredict(validation_label_4, validation_instance_4, RBFmodel_3);
        [predict_label, RBF_accuracy4, dec_values] = svmpredict(validation_label_4, validation_instance_4, RBFmodel_5);
        accuracy4 = (RBF_accuracy1(1) + RBF_accuracy2(1) + RBF_accuracy3(1) + RBF_accuracy4(1))/4;
        
        [predict_label, RBF_accuracy1, dec_values] = svmpredict(validation_label_5, validation_instance_5, RBFmodel_1);
        [predict_label, RBF_accuracy2, dec_values] = svmpredict(validation_label_5, validation_instance_5, RBFmodel_2);
        [predict_label, RBF_accuracy3, dec_values] = svmpredict(validation_label_5, validation_instance_5, RBFmodel_3);
        [predict_label, RBF_accuracy4, dec_values] = svmpredict(validation_label_5, validation_instance_5, RBFmodel_4); 
        accuracy5 = (RBF_accuracy1(1) + RBF_accuracy2(1) + RBF_accuracy3(1) + RBF_accuracy4(1))/4;
        
        RBF_accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4 + accuracy5)/5;
        accuracy_matrix(i,j) = RBF_accuracy;
    end
end

% Find best C and Alpha values
max_accuracy = 0;
max_accuracy_C = 0;
max_accuracy_ALPHA = 0;
for i = 1:13
    for j = 1:13
        if (accuracy_matrix(i,j) > max_accuracy)
              max_accuracy = accuracy_matrix(i,j);
              max_accuracy_ALPHA = j;
              max_accuracy_C = i;
        end
    end
end

best_alpha = ALPHA(max_accuracy_ALPHA);
best_C = C(max_accuracy_C);
disp("Best accuracy:");
disp(max_accuracy);
disp("Best alpha value:");
disp(best_alpha);
disp("Best C value:");
disp(best_C);
disp("Accuracy matrix:");
disp(accuracy_matrix);

% Train an RBF kernel SVM with best C and alpha values
options_string = '-c';
options_string = strcat({options_string}, {' '}, {num2str(best_C)});
options_string = strcat(options_string, {' '}, '-g');
options_string = strcat(options_string, {' '}, {num2str(best_alpha)});

options_string = char(options_string);

% Classify test set and check accuracy
optimal_model = svmtrain(train_label_vector, train_instance_matrix, options_string);
[predict_label, accuracy, dec_values] = svmpredict(test_label_vector, test_instance_matrix, optimal_model);

	