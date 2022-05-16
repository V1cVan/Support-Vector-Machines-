load logmap.mat

% orderlist = [2 5 10 20 40 80 100 120 140];
orderlist = [1 2 5 10 20 50 80 100];
n = length(orderlist);

gamlist = zeros(1,n);
sig2list = zeros(1,n);

RMSE_order = zeros(1,n);
MAE_order = zeros(1,n);

for i = 1:n
    
    order = orderlist(i);
    
    % Prepare the training data
    X = windowize(Z, 1:(order + 1));    
    Y = X(:, end);
    X = X(:, 1:order);
    
    % tune the gam and sig2 with training data
    model = { X , Y , 'f', [], [],'RBF_kernel'};
    algorithm = 'simplex';
    [gam , sig2, cost] = tunelssvm(model, algorithm, 'crossvalidatelssvm',{10, 'mse'});
    
    % notedown the tuned gam and sig2 for each order number 
    gamlist(i) = gam;
    sig2list(i) = sig2;
    
    
    % train the model with tuned gam and sig2
    [alpha, b] = trainlssvm({X, Y, 'f', gam, sig2,'RBF_kernel'});
    
    % prepare the test data for prediction
    Xs = Z(end - order + 1:end , 1);
    nb = 50;
    
    % make the prediction and calucate the prediction performance with MSE
    % and correlation;
    prediction = predict({X, Y, 'f', gam, sig2,'RBF_kernel'}, Xs, nb);
    
%     figure;
%     
%     hold on;
%     plot_title = ['Prediction with order: ' num2str(order) ' gam: ' num2str(gam) ' sig2: ' num2str(sig2)];
%     title(plot_title);
%     plot(Ztest, 'k');
%     plot(prediction, 'r');
%     hold off;
    
    RMSE_order(i) = sqrt(mean((Ztest-prediction).^2));
    
    MAE_order(i) = mean(abs((Ztest-prediction)));
    
%     R = corrcoef(Ztest,prediction);
%     R_order(i) = R(1,2);
%     

end 

% plot the order vs MSE and Correlation 
figure 
hold on
xlabel('Order')
ylabel('MSE and MAE')
plot(orderlist, RMSE_order,'r');
plot(orderlist, MAE_order,'b');
legend('RMSE','MAE')
hold off
% 
% % [alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});
% 
% [min_mse,index] = min(MSE_order);
% disp(min_mse);
% 

index = 6;
opt_order = orderlist(index);
disp(opt_order);
opt_gam = gamlist(index);
disp(opt_gam);
opt_sig2 = sig2list(index);
disp(opt_sig2);
% 
% 
% 
Xs = Z(end - opt_order + 1:end , 1);

X = windowize(Z, 1:(opt_order + 1));    
Y = X(:, end);
X = X(:, 1:opt_order);

nb = 50;
prediction = predict({X, Y, 'f', opt_gam, opt_sig2,'RBF_kernel'}, Xs, nb);

figure;

hold on;
plot_title = ['Prediction with order: ' num2str(opt_order) ' gam: ' num2str(opt_gam) ' sig2: ' num2str(opt_sig2)];
title(plot_title);
plot(Ztest, 'k');
plot(prediction, 'r');
hold off;

% MSE = immse(Ztest,prediction);
% disp(MSE);


