% Create training data 
X = 2.*rand(30,2)-1;
Y = sign(sin(X(:,1))+X(:,2));

% Simple classification task 
gam = 10; % regularisation parameter - tradeoff between fitting minimisation error and smoothness 
sig2 = 0.2; % For rbf sigma^2 is the badwidth 
type = 'classification';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

% Evaluate new points for this model (inference) 
Xt = 2.*rand(10,2)-1;
Ytest = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);

% Plotting of the results 
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
% plotlssvm({Xt,Ytest,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
