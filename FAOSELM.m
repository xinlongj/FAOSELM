function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = FAOSELM(TrainingData, IncrementData, TestingData, FA_Type, FeatureChangeDimension, Elm_Type, nHiddenNeurons, ActivationFunction, Block)

% Usage: FAOSELM(TrainingData, IncrementData, TestingData, FA_Type, FeatureChangeDimension, Elm_Type, nHiddenNeurons, ActivationFunction, Block)
% OR: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = FAOSELM(TrainingData, IncrementData, TestingData, FA_Type, FeatureChangeDimension, Elm_Type, nHiddenNeurons, ActivationFunction, Block)
%
% Input:
% TrainingData           - Training data set
% IncrementData          - Increment data set
% TestingData            - Filename of testing data set
% FA_Type                - 0 for feature reduction; 1 for feature increment
% FeatureChangeDimension - A vector used to show the feature dimension change
% Elm_Type               - 0 for regression; 1 for (both binary and multi-classes) classification
% nHiddenNeurons         - Number of hidden neurons assigned to the FAOSELM
% ActivationFunction     - Type of activation function:
%                           'rbf' for radial basis function, G(a,b,x) = exp(-b||x-a||^2)
%                           'sig' for sigmoidal function, G(a,b,x) = 1/(1+exp(-(ax+b)))
%                           'sin' for sine function, G(a,b,x) = sin(ax+b)
%                           'hardlim' for hardlim function, G(a,b,x) = hardlim(ax+b)
% Block                  - Size of block of data learned by FAOSELM in each step
%
% Output: 
% TrainingTime           - Time (seconds) spent on training FAOSELM
% TestingTime            - Time (seconds) spent on predicting all testing data
% TrainingAccuracy       - Training accuracy: 
%                           RMSE for regression or correct classification rate for classifcation
% TestingAccuracy        - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classifcation
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Samples:
% Feature reduction: FAOSELM(TrainData, IncreData, TestData, 0, [3 5], 1, 25, 'rbf', 100)
% Feature increment: FAOSELM(TrainData, IncreData, TestData, 1, [2 8], 1, 25, 'rbf', 200)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Authors:Xinlon Jiang    
%    Affiliate: Institute of Computing Technology, CAS
%    EMAIL:jiangxinlong@ict.ac.cn
%    Paper: ¡¶Feature Adaptive Online Sequential Extreme Learning Machine for lifelong indoor localization¡·
%    Website£ºhttp://dl.acm.org/citation.cfm?id=2877088
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Macro definition
ELM_REGRESSION = 0;
ELM_CLASSIFICATION = 1;

FA_REDUCTION = 0;
FA_INCREMENT = 1;

%%%%%%%%%%% Load dataset
training_data = TrainingData;
increment_data = IncrementData;
testing_data = TestingData;

T = training_data(:,1); 
P = training_data(:,2:size(training_data,2));

IN.T = increment_data(:,1);
IN.P = increment_data(:,2:size(increment_data,2));

TV.T = testing_data(:,1); 
TV.P = testing_data(:,2:size(testing_data,2));

clear training_data increment_data testing_data;

%%%%%%%%%%% number of training samples, increment samples and testing samples
nTrainingData  = size(P,1);
nIncrementData = size(IN.P,1);
nTestingData   = size(TV.P,1);

%%%%%%%%%%%% Preprocessing T in the case of CLASSIFICATION 
if Elm_Type == ELM_CLASSIFICATION
    sorted_target = sort(cat(1,cat(1,T,IN.T),TV.T),1);
    label = zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1) = sorted_target(1,1);
    j = 1;
    for i = 2:(nTrainingData + nIncrementData + nTestingData)
        if sorted_target(i,1) ~= label(j,1)
            j = j + 1;
            label(j,1) = sorted_target(i,1);
        end
    end
    nClass = j;
    nOutputNeurons = nClass;

    %%%%%%%%%% Processing the targets of training samples
    temp_T = zeros(nTrainingData,nClass);
    for i = 1:nTrainingData
        for j = 1:nClass
            if label(j,1) == T(i,1)
                break; 
            end
        end
        temp_T(i,j) = 1;
    end
    T = temp_T * 2 - 1;

    %%%%%%%%%% Processing the targets of increment samples
    temp_T = zeros(nIncrementData,nClass);
    for i = 1:nIncrementData
        for j = 1:nClass
            if label(j,1) == IN.T(i,1)
                break; 
            end
        end
        temp_T(i,j) = 1;
    end
    IN.T = temp_T * 2 - 1;
    %%%%%%%%%% Processing the targets of testing samples
    temp_TV_T = zeros(nTestingData,nClass);
    for i = 1:nTestingData
        for j = 1:nClass
            if label(j,1) == TV.T(i,1)
                break; 
            end
        end
        temp_TV_T(i,j) = 1;
    end
    TV.T = temp_TV_T * 2 - 1;
end
clear temp_T temp_TV_T sorted_target

start_time_train = cputime;

if FA_Type == FA_REDUCTION
    nTrainingInputNeurons = size(P,2);
    Tr = eye(nTrainingInputNeurons); % Tr is used for input-weight transfer
    Tr(:,FeatureChangeDimension) = [];
    TRAIN.IW = rand(nHiddenNeurons, nTrainingInputNeurons) * 2 - 1;
    INCRESE.IW = TRAIN.IW * Tr;
    TEST.IW =  INCRESE.IW;
end

if FA_Type == FA_INCREMENT
    nIncrementInputNeurons = size(IN.P,2);
    Tr = eye(nIncrementInputNeurons); % Tr is used for input-weight transfer
    Tr(:,FeatureChangeDimension) = [];
    INCRESE.IW = rand(nHiddenNeurons, nIncrementInputNeurons) * 2 - 1;
    TEST.IW =  INCRESE.IW;
    TRAIN.IW = INCRESE.IW * Tr;
    
end

%%%%%%%%%%% step 1 Initialization Phase
switch lower(ActivationFunction)
    case{'rbf'}
        Bias = rand(1,nHiddenNeurons);
        H0 = RBFun(P,TRAIN.IW,Bias);
    case{'sig'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = SigActFun(P,TRAIN.IW,Bias);
    case{'sin'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = SinActFun(P,TRAIN.IW,Bias);
    case{'hardlim'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = HardlimActFun(P,TRAIN.IW,Bias);
        H0 = double(H0);
end
M = pinv(H0' * H0);
beta = pinv(H0) * T;
clear H0;

%-----------------------------------------------------------%
%   The iterative formula is                                %
%   K1 = H1T * H1                                           %
%   beta1 = beta0 + K1' * H1T * £¨T1 - H1 * beta0£©         % 
%-----------------------------------------------------------%

%%%%%%%%%%%%% step 2 Sequential Learning Phase
for n = 1 : Block : nIncrementData
    if (n+Block-1) > nIncrementData
        Tn = IN.T(n:nIncrementData,:);
        Pn = IN.P(n:nIncrementData,:);

        Block = size(Pn,1);             %%%% correct the block size
        clear V;                        %%%% correct the first dimention of V 
    else
        Tn = IN.T(n:(n+Block-1),:);
        Pn = IN.P(n:(n+Block-1),:);
    end
    switch lower(ActivationFunction)
        case{'rbf'}
            H = RBFun(Pn,INCRESE.IW,Bias);
        case{'sig'}
            H = SigActFun(Pn,INCRESE.IW,Bias);
        case{'sin'}
            H = SinActFun(Pn,INCRESE.IW,Bias);
        case{'hardlim'}
            H = HardlimActFun(Pn,INCRESE.IW,Bias);
    end
    M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M; 
    beta = beta + M * H' * (Tn - H * beta);
end
end_time_train = cputime;
TrainingTime = end_time_train - start_time_train        
clear Pn Tn H M;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch lower(ActivationFunction)
    case{'rbf'}
        HTrain = RBFun(P, TRAIN.IW, Bias);
    case{'sig'}
        HTrain = SigActFun(P, TRAIN.IW, Bias);
    case{'sin'}
        HTrain = SinActFun(P, TRAIN.IW, Bias);
    case{'hardlim'}
        HTrain = HardlimActFun(P, TRAIN.IW, Bias);
end
Y = HTrain * beta;

clear HTrain;

%%%%%%%%%%% Performance Evaluation
start_time_test = cputime; 
switch lower(ActivationFunction)
    case{'rbf'}
        HTest = RBFun(TV.P, TEST.IW, Bias);
    case{'sig'}
        HTest = SigActFun(TV.P, TEST.IW, Bias);
    case{'sin'}
        HTest = SinActFun(TV.P, TEST.IW, Bias);
    case{'hardlim'}
        HTest = HardlimActFun(TV.P, TEST.IW, Bias);
end    
TY = HTest * beta;
clear HTest;
end_time_test = cputime;
TestingTime = end_time_test - start_time_test  

if Elm_Type == ELM_REGRESSION
    %%%%%%%%%%%%%% Calculate RMSE in the case of REGRESSION
    TrainingAccuracy = sqrt(mse(T - Y))
    TestingAccuracy  = sqrt(mse(TV.T - TY))
    
elseif Elm_Type == ELM_CLASSIFICATION
	%%%%%%%%%% Calculate correct classification rate in the case of CLASSIFICATION
    MissClassificationRate_Training = 0;
    MissClassificationRate_Testing = 0;
    
    for i = 1 : nTrainingData
        [x, label_index_expected] = max(T(i,:));
        [x, label_index_actual] = max(Y(i,:));
        if label_index_actual ~= label_index_expected
            MissClassificationRate_Training = MissClassificationRate_Training + 1;
        end
    end
    TrainingAccuracy = 1 - MissClassificationRate_Training / nTrainingData
    
    for i = 1 : nTestingData
        [x, label_index_expected] = max(TV.T(i,:));
        [x, label_index_actual] = max(TY(i,:));
        if label_index_actual ~= label_index_expected
            MissClassificationRate_Testing = MissClassificationRate_Testing + 1;
        end
    end
    TestingAccuracy = 1 - MissClassificationRate_Testing / nTestingData  
end