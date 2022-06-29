clc; clear; close all 
load iris

% ======================== 1.3.1 ==========================================


%% LS-SVM classifier - Linear Kernel 
type='c'; % Classification 
gamma = 1; % Regularisation 
disp('Linear kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamma,[],'lin_kernel'});

figure; 
plotlssvm({Xtrain,Ytrain,type,gamma,[],'lin_kernel','preprocess'},{alpha,b});
grid on 
fig = gcf;
exportgraphics(fig, '../figures/linear_classification.pdf', 'ContentType', 'vector', 'Resolution', 300);
[Ypred, Zt] = simlssvm({Xtrain,Ytrain,type,gamma,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Ypred~=Ytest); 
fprintf('\n On Test Set (linear Kernel): \n Misclassification Number = %d, Error percentage = %.2f%%\n', err, err/length(Ytest)*100)

%% LS-SVM classifier - Polynomial Kernel 
type='c'; 
gamma = 1; 
t = 1; % Poly intercept 
acc_list = []; 
degree_list = [1,2,3,4,5];
for degree = degree_list
    disp(fprintf('\n == \n Polynomial kernel of degree %d',degree))   
    
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamma,[t; degree],'poly_kernel'});
      
    figure; 
    plotlssvm({Xtrain,Ytrain,type,gamma,[t; degree],'poly_kernel','preprocess'},{alpha,b});
    grid on 
    fig = gcf;
    figname = sprintf('../figures/polynomial_classif_deg_%d.pdf',degree);
    exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);
    [Ypred, Zt] = simlssvm({Xtrain,Ytrain,type,gamma,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);
    
    num_errors = sum(Ypred~=Ytest); 
    perc_errors = num_errors/length(Ytest)*100; 
    acc_list = [acc_list, 100-perc_errors]; 
    fprintf('\n Polynomial kernel with degree = %d:\nNumber missclassifications = %d, Error percentage = %.2f%%\n', degree, num_errors, perc_errors)
end

figure; plot(degree_list,acc_list, '*-','linewidth', 2); 
% xlim([1,degree_list(end)]);
ylim([0,100]);
grid on;
xlabel('Polynomial degree') 
ylabel('Classificaiton accuracy [%]') 
title('Classification accuracy for increasing polynomial degrees') ;
fig = gcf;
exportgraphics(fig, '../figures/class_acc_poly_deg_val.pdf', 'ContentType', 'vector', 'Resolution', 300);

%% LS-SVM classifier - RBF Kernel 
close all 
disp('RBF kernel')
gamma = 1; 
sig2list=[0.01, 0.1, 1, 5, 10, 25];

acc_list = []; 
err_list = [];
for sigma2=sig2list
    disp(['gam : ', num2str(gamma), '   sig2 : ', num2str(sigma2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamma,sigma2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gamma,sigma2,'RBF_kernel','preprocess'},{alpha,b});
    grid on
    fig = gcf;
    figname = sprintf('../figures/rbf_classif_sigma2_%.2f.pdf',sigma2);
    exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);

    % Obtain the output of the trained classifier
    [Ypred, Zt] = simlssvm({Xtrain,Ytrain,type,gamma,sigma2,'RBF_kernel'}, {alpha,b}, Xtest);
    
    num_err = sum(Ypred~=Ytest); 
    perc_err = num_err/length(Ytest)*100;
    err_list=[err_list; perc_err];
    acc_list = [acc_list; 100-perc_err];
    
    fprintf('\n RBF kernel for sigma^2 = %d:\n #Number of missclassifications = %d, Percentage errors = %.2f%% \n', sigma2, num_err, perc_err)  
end


%
% make a plot of the misclassification rate wrt. sig2
%
% figure;
% plot(log(sig2list), err_list, '*-'), 
% xlabel('log(sig2)'), ylabel('number of misclass'),

figure; plot(log(sig2list), acc_list, '*-', 'linewidth', 2); 
ylim([0,100]);
xlabel('log(/sigma^2)') 
ylabel('Classificaiton accuracy [%]') 
grid on;
title('Classification accuracy for increasing \sigma^2 values') ;
fig = gcf;
exportgraphics(fig, '../figures/class_acc_sigma2_val.pdf', 'ContentType', 'vector', 'Resolution', 300);

% ======================== 1.3.2 ==========================================

%% Tuning paramerts using validation techniques
%{
k-fold cross validation: 
The data is once permutated randomly, then it is divided into L (by default 10)
  disjoint sets. In the i-th (i=1,...,l) iteration, the i-th set is used to estimate
  the performance ('validation set') of the model trained on the other l-1 sets ('training set').
  Finally, the l (denoted by L) different estimates of the performance are combined (by default by the 'mean').
  The assumption is made that the input data are distributed independent and identically over the
  input space.

Leave-one-out cross validation: 
In each iteration, one leaves one point, and fits a model on the
  other data points. The performance of the model is estimated
  based on the point left out. This procedure is repeated for each
  data point. Finally, all the different estimates of the
  performance are combined (default by computing the mean). The
  assumption is made that the input data is distributed independent
  and identically over the input space.
%}


values = [1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4 1e5];
values = logspace(-3,3,20);
gamma_list = values;
sigma2_list = values;

gamma_len = length(gamma_list);
sigma_len = length(sigma2_list);

perf_mat_split = zeros(gamma_len, sigma_len);
perf_mat_cv = zeros(gamma_len, sigma_len);
perf_mat_loo = zeros(gamma_len, sigma_len);

for idx_gamma = [1:1:gamma_len]
    for idx_sigma2 = [1:1:sigma_len]
        gamma = gamma_list(idx_gamma);
        sigma2 = sigma2_list(idx_sigma2);
        
        model = {Xtrain,Ytrain,'c', gamma, sigma2,'RBF_kernel'};
        measure = 'misclass';
        
        % Random split 
        random_split_ratio = 0.80; % Split between training and validation 
        perf_split = rsplitvalidate(model, random_split_ratio, measure);
        perf_mat_split(idx_gamma, idx_sigma2) = perf_split;
        
        % K-fold cross validation
        n_folds = 10; % Number of folds used in the cross validation procedure 
        pref_cv = crossvalidate(model, n_folds, measure);
        perf_mat_cv(idx_gamma, idx_sigma2) = pref_cv;
        
        % Leave-one-out validation
        perf_loo = leaveoneout(model, measure);
        perf_mat_loo(idx_gamma, idx_sigma2) = perf_loo;
    end
end


[x, y] = meshgrid(log(gamma_list), log(sigma2_list)); 

figure;
surf(x, y, perf_mat_split)
colorbar
view(-135,45)
xlabel('log10(\gamma)')
ylabel('log10(\sigma^2)')
zlabel('Misclassification cost')
title('Random split (80/20)')
grid on 
fig = gcf;
exportgraphics(fig, '../figures/random_split_validation_surf.pdf', 'ContentType', 'vector', 'Resolution', 300);

figure; 
surf(x, y, perf_mat_cv)
colorbar
view(-135,45)
xlabel('log10(\gamma)')
ylabel('log10(\sigma^2)')
zlabel('Misclassification cost')
title('10-fold Cross Validation')
grid on 
fig = gcf;
exportgraphics(fig, '../figures/10_fold_cross_validation_surf.pdf', 'ContentType', 'vector', 'Resolution', 300);


figure; 
surf(x, y, perf_mat_loo)
colorbar
view(-135,45)
xlabel('log10(\gamma)') 
ylabel('log10(\sigma^2)')
zlabel('Misclassification cost')
title('Leave-one-out Cross Validation')
grid on 
fig = gcf;
exportgraphics(fig, '../figures/loo_cross_validation_surf.pdf', 'ContentType', 'vector', 'Resolution', 300);


% ======================== 1.3.3 ==========================================

%% Automatic parameter tuning 
% Two algorithms exists - Simplex (Nelder-Mead method) and Gridsearch
% (brute force) 
clc
text_output = [];
gammas = [] ;
sigma2s = [] ;
costs = [];
execution_speeds = []; 

for loop = [1:1]
    for algo_idx = [1:2]
        algo_idx
        if algo_idx == 1
            algo = 'simplex';
            display(algo);
        else 
            algo = 'gridsearch';
            display(algo);
        end
        tic = cputime
        [gam ,sig2 , cost ] = tunelssvm({ Xtrain , Ytrain , 'c', [], [], 'RBF_kernel'}, algo, 'crossvalidatelssvm',{10, 'misclass'});
        toc = cputime-tic
        gammas = [gammas, gam];
        sigma2s = [sigma2s, sig2]; 
        costs = [cost, costs]; 
        execution_speeds = [execution_speeds, toc];
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
        if algo_idx == 1
            % simp
            plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
            grid on
            fig = gcf;
            figname = '../figures/rbf_simplex_optimal.pdf';
            exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);
            
        else 
            % grid
            plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
            grid on
            fig = gcf;
            figname = '../figures/rbf_gridsearch_optimal.pdf';
            exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);
            
        end
        
    end

    simplex = sprintf('Simplex Tuning Results (Run[%d]):\n  Gamma = %.2f\n  Sigma^2 = %.2f\n  Cost = %.2f\n  CPUtime = %.2f\n', loop, gammas(1), sigma2s(1), costs(1), execution_speeds(1));
    gridsearch = sprintf('Gridsearch Tuning Results (Run[%d]):\n  Gamma = %.2f\n  Sigma^2 = %.2f\n  Cost = %.2f\n  CPUtime = %.2f\n', loop, gammas(2), sigma2s(2), costs(2), execution_speeds(2));

    text_output = [text_output, simplex, gridsearch];
    
end
text_output

% ======================== 1.3.4 ==========================================

%% ROC Curves 
%{ 
help roc: 
The roc curve shows the separation abilities of a binary
  classifier: by iteratively setting the possible classifier
  thresholds, the dataset is tested on misclassifications. As a
  result, a plot is shown where the various outcomes are
  described. If the plot has a surface of 1 on test data, a
  perfectly separating classifier is found (on that particular
  dataset), if the area equals 0.5, the classifier has no
  discriminative power at all. In general, this function can be
  called with the latent variables Zt and the corresponding class labels Yclass
%}
clear,clc,close all 
load iris.mat

for algo_idx = [1:2]
    % Tune the paramerters of the algorithm
    if algo_idx == 1
        algo = 'simplex'
    else 
        algo = 'gridsearch'
    end 
    model = { Xtrain , Ytrain , 'c', [], [], 'RBF_kernel'};
    [gam, sig2 ,cost ] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
    tuned_model = {Xtrain , Ytrain, 'c', gam , sig2 ,'RBF_kernel'};
    
    % Train the classification model.
    [alpha , b] = trainlssvm (tuned_model);
    % Classification of the test data.
    [Ypred , Ylatent] = simlssvm (tuned_model, {alpha , b}, Xtest);
    % Generating the ROC curve.
    figure()
    area = roc( Ylatent , Ytest);
    grid on 
    if algo_idx == 1
        sprintf('Area under simplex optimised classifier = %.2f', area)
        fig = gcf;
        exportgraphics(fig, '../figures/simplex_rbf_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);
    else 
        sprintf('Area under gridsearch optimised classifier = %.2f', area)
        fig = gcf;
        exportgraphics(fig, '../figures/gridsearch_rbf_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);
    end 
end

% Untuned model with a linear kernel 
gam = 1 ;
sig2 = [];
untuned_model = {Xtrain, Ytrain, 'c', gam,sig2,'lin_kernel'};
% Train the classification model.
[alpha,b] = trainlssvm(untuned_model);
% Classification of the test data.
[Ypred , Ylatent] = simlssvm (untuned_model, {alpha , b}, Xtest );
% Generating the ROC curve.
figure();
lin_class_results = roc( Ylatent , Ytest);
grid on
fig = gcf;
exportgraphics(fig, '../figures/linear_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);
sprintf("Linear classifier area under ROC curve = %.2f", lin_class_results(1)) 


% ======================== 1.3.5 ==========================================

%% Bayesian framework  
%{
help bay_modoutClass
Using a Bayesian framework it is possible to get probability estimates. 
Estimate the posterior class probabilities of a binary classifier using Bayesian inference

  >> [Ppos, Pneg] = bay_modoutClass({X,Y,'classifier',gam,sig2}, Xt)
  >> [Ppos, Pneg] = bay_modoutClass(model, Xt)
  
  Calculate the probability that a point will belong to the
  positive or negative classes taking into account the uncertainty
  of the parameters. Optionally, one can express prior knowledge as
  a probability between 0 and 1, where prior equal to 2/3 means
  that the  prior positive class probability is 2/3 (more likely to
  occur than the negative class).
  For binary classification tasks with a 2 dimensional input space,
  one can make a surface plot by replacing Xt by the string 'figure'.
%}
clear,clc,close all 
load iris.mat
gam = 1; 
sig2 = 1;
figure()
bay_modoutClass({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');
grid on
colorbar
fig = gcf;
figname = sprintf('../figures/bayes_rbf_gamma_%d_sig2_%d.pdf', gam, sig2);
exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);

% Investigate different values for gamma 
gammas = [0.1, 1, 10];
sig2 = 1; 
for gam = gammas
    figure()
    bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');
    grid on
    colorbar
    fig = gcf;
    figname = sprintf('../figures/bayes_rbf_gamma_%d_sig2_%d.pdf', gam, sig2);
    exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);
end 



% Investigate different values for sigma2
sigma2s = [0.1, 1, 10];
gam = 1; 
for sig2 = sigma2s
    figure()
    bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');
    grid on
    colorbar
    fig = gcf;
    figname = sprintf('../figures/bayes_rbf_gamma_%d_sig2_%d.pdf', gam, sig2);
    exportgraphics(fig, figname, 'ContentType', 'vector', 'Resolution', 300);
end 

% ======================== 2.1 ==========================================

%% Ripley dataset - Homework problems  
%{
    The well-known Ripley dataset problem consists of two classes where the 
    data for each class have been generated by a mixture of two Gaussian distributions
%}
clc, clear, close all 
disp('================ RIPLEY DATASET ================') 
load '../data/ripley.mat'

X = [Xtrain;Xtest];
Y = [Ytrain;Ytest];
data_shape = size(X)
data_mean = mean(X)
data_std = std(X)

figure()
scatter(X(:,1),X(:,2),25,Y,'filled') 
grid on 
xlabel('X1')
ylabel('X2')
fig = gcf;
exportgraphics(fig, '../figures/ripley_data.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- RBF KERNEL -------------------') 
% Tune the paramerters of the algorithm
algo = 'gridsearch';
kernel = 'RBF_kernel';
model = {Xtrain , Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/ripley_simplex_rbf_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised rbf classifier on Ripley dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/ripley_simplex_rbf_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- POLYNOMIAL KERNEL -------------------') 
% Tune the paramerters of the algorithm
kernel = 'poly_kernel';
model = {Xtrain, Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/ripley_simplex_polynomial_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised polynomial classifier on Ripley dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/ripley_simplex_polynomial_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- LINEAR KERNEL -------------------') 
% Tune the paramerters of the algorithm
kernel = 'lin_kernel';
model = {Xtrain, Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/ripley_simplex_linear_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised linear classifier on Ripley dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/ripley_simplex_linear_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);

% ======================== 2.2 ==========================================

%% Wisconsin Breast Cancer Dataset - Homework problems  
%{
    https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
%}

clc, clear, close all 
disp('================ BREAST CANCER DATASET ================') 
load '../data/breast.mat'

% Convert data into standard form
Xtrain = trainset; 
Ytrain = labels_train; 
Xtest = testset; 
Ytest = labels_test;

X = [Xtrain;Xtest];
Y = [Ytrain;Ytest];
data_shape = size(X)
data_mean = mean(X)
data_std = std(X)
data_corr = corr(X)

figure()
rng('default') % for fair comparison
X_plot = tsne(X);
gscatter(X_plot(:,1),X_plot(:,2),Y) 
xlabel('X1')
ylabel('X2')
grid on 
fig = gcf;
exportgraphics(fig, '../figures/breast_data.pdf', 'ContentType', 'vector', 'Resolution', 300);

% figure() 
% imagesc(data_corr)
% set(gca, 'XTick', 1:sizeofyourcorrmatrix); % center x-axis ticks on bins
% set(gca, 'YTick', 1:nsizeofyourcorrmatrix); % center y-axis ticks on bins
% set(gca, 'XTickLabel', yourlabelnames); % set x-axis labels
% set(gca, 'YTickLabel', yourlabelnames); % set y-axis labels
% title('Correlation Heatmap Breast Cancer Dataset', 'FontSize', 12); % set title
% colormap('jet'); % Choose jet or any other color scheme
% colorbar on; % 


disp('------------------- RBF KERNEL -------------------') 
% Tune the paramerters of the algorithm
algo = 'simplex';
kernel = 'RBF_kernel';
model = {Xtrain , Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/breast_rbf_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised rbf classifier on Breast Cancer dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/breast_rbf_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- POLYNOMIAL KERNEL -------------------') 
% Tune the paramerters of the algorithm
kernel = 'poly_kernel';
model = {Xtrain, Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/breast_polynomial_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised polynomial classifier on Breast cancer dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/breast_polynomial_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- LINEAR KERNEL -------------------') 
% Tune the paramerters of the algorithm
kernel = 'lin_kernel';
model = {Xtrain, Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/breast_linear_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised linear classifier on Breast cancer dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/breast_linear_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);





% ======================== 2.3 ==========================================

%% Diabetes Dataset - Homework problems  
%{
    https://archive.ics.uci.edu/ml/datasets/diabetes
%}

clc, clear, close all 
disp('================ DIABETES DATASET ================') 
load '../data/diabetes.mat'

% Convert data into standard form
Xtrain = trainset; 
Ytrain = labels_train; 
Xtest = testset; 
Ytest = labels_test;

X = total;
Y = labels_total;
data_shape = size(X)
data_mean = mean(X)
data_std = std(X)
data_corr = corr(X)

figure()
rng('default') % for fair comparison
X_plot = tsne(X);
gscatter(X_plot(:,1),X_plot(:,2),Y) 
xlabel('X1')
ylabel('X2')
grid on 
fig = gcf;
exportgraphics(fig, '../figures/diabetes_data.pdf', 'ContentType', 'vector', 'Resolution', 300);


% figure() 
% imagesc(data_corr)
% set(gca, 'XTick', 1:sizeofyourcorrmatrix); % center x-axis ticks on bins
% set(gca, 'YTick', 1:nsizeofyourcorrmatrix); % center y-axis ticks on bins
% set(gca, 'XTickLabel', yourlabelnames); % set x-axis labels
% set(gca, 'YTickLabel', yourlabelnames); % set y-axis labels
% title('Correlation Heatmap Breast Cancer Dataset', 'FontSize', 12); % set title
% colormap('jet'); % Choose jet or any other color scheme
% colorbar on; % 


disp('------------------- RBF KERNEL -------------------') 
% Tune the paramerters of the algorithm
algo = 'simplex'
kernel = 'RBF_kernel';
model = {Xtrain , Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/diabetes_rbf_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised rbf classifier on Diabetes dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/diabetes_rbf_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- POLYNOMIAL KERNEL -------------------') 
% Tune the paramerters of the algorithm
kernel = 'poly_kernel';
model = {Xtrain, Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/diabetes_polynomial_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised polynomial classifier on Diabetes dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/diabetes_polynomial_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


disp('------------------- LINEAR KERNEL -------------------') 
% Tune the paramerters of the algorithm
kernel = 'lin_kernel';
model = {Xtrain, Ytrain, 'c', [], [], kernel};
[gam, sig2, cost] = tunelssvm(model, algo, 'crossvalidatelssvm',{10, 'misclass'});
tuned_model = {Xtrain , Ytrain, 'c', gam, sig2, kernel};

% Train the classification model.
[alpha ,b] = trainlssvm (tuned_model);

% Plot classification result
figure()
plotlssvm(tuned_model, {alpha, b}); 
grid on 
colorbar
fig = gcf;
exportgraphics(fig, '../figures/diabetes_linear_classification_result.pdf', 'ContentType', 'vector', 'Resolution', 300);

% Classification on the test data.
[Ypred , Ylatent] = simlssvm(tuned_model, {alpha , b}, Xtest);

% Generating the ROC curve.
figure()
area = roc(Ylatent , Ytest);
grid on 
sprintf('Area under simplex optimised linear classifier on Diabetes dataset = %.3f', area)
fig = gcf;
exportgraphics(fig, '../figures/diabetes_linear_classifier_roc.pdf', 'ContentType', 'vector', 'Resolution', 300);


