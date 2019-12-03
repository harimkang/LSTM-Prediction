%	Date : 2019.12.03
%	Programmer : Harim Kang
%	Description : Time series data prediction using LSTM Net

%	Data Cleaning : 2015.01~2019.10 Korean jeans online sale average
%	Data Ref : https://www.data.go.kr/dataset/15004449/fileData.do
%	Use data from only the daily average jeans price from the online
%	collection price data from that source. (jean_sales.xlsx)
%	See extract_jean_sale_average.m for Extraction Average Data

jean_data = readtable('jean_sales.xlsx');

%	Fill the NaN value with the Nearest value.
jean_data.sales_price = fillmissing(jean_data.sales_price, 'nearest');
lenofdata = length(jean_data.sales_price);


for i=1 : length(jean_data.collect_day)
    jean_data.collect_day(i) = strip(jean_data.collect_day(i),"'");
end

Y = jean_data.sales_price;
data = Y';

%   2015.01.01 ~ 2019.05.06 (90%) : Training Data Set
%   2019.05.07 ~ 2019.10.31 (10%) : Test Data Set
numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

%   Normalize sales_price to a value between 0 and 1 (Training Data Set)
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%LSTM Net Architecture Def
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
	
%	Train LSTM Net
net = trainNetwork(XTrain,YTrain,layers,options);

%	Normalize sales_price to a value between 0 and 1 (Testing Data Set)
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

%   Predict as long as the test period (2019.05.07 ~ 2019.10.31)
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%   RMSE calculation of test data set
YTest = dataTest(2:end);
YTest = (YTest - mu) / sig;
rmse = sqrt(mean((YPred-YTest).^2))

%	Denormalize Data
YPred = sig*YPred + mu;
YTest = sig*YTest + mu;

%   X Label : Collect Day
x_data = datetime(jean_data.collect_day);
x_train = x_data(1:numTimeStepsTrain+1);
x_train = x_train';
x_pred = x_data(numTimeStepsTrain:numTimeStepsTrain+numTimeStepsTest);

%   Train + Predict Plot
figure
plot(x_train(1:end-1),dataTrain(1:end-1))
hold on
plot(x_pred,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Collect Day")
ylabel("Sales Price")
title("Forecast")
legend(["Observed" "Forecast"])

%  RMSE Plot : Test + Predict Plot
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Sales Price")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Collect Day")
ylabel("Error")
title("RMSE = " + rmse)

%   Train + Test + Predict Plot
figure
plot(x_data,Y)
hold on
plot(x_pred,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Collect Day")
ylabel("Sales Price")
title("Compare Data")
legend(["Raw" "Forecast"])