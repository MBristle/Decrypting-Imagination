%Playground
%% SVM - prepare 

% Preperation of training data
minL= min(cell2mat(vertcat(dataset.all(:,6)))); % ,dataset.all(:,7)
dataset.X={};
dataset.Y={};
for i=1:length(dataset.all(:,1))
    dataset.X=vertcat(dataset.X,[...
        dataset.all{i,10}(1,1:minL),... pupil
        dataset.all{i,11}(1,1:minL),... duration
        dataset.all{i,12}(1,1:minL),... X
        dataset.all{i,13}(1,1:minL)... Y
        ]);
end
dataset.X=cell2mat(dataset.X);
%dataset.X=rescale(dataset.X);


% lables
for i=1:length(dataset.all(:,1))
    switch dataset.all{i,8}
        case 'art '
            dataset.Y{i,1}=true;
            dataset.Y{i,2}=false;
            dataset.Y{i,3}=false;
        case 'face'
            dataset.Y{i,1}=false;
            dataset.Y{i,2}=true;
            dataset.Y{i,3}=false;
        case 'land'
            dataset.Y{i,1}=false;
            dataset.Y{i,2}=false;
            dataset.Y{i,3}=true;
    end
end
dataset.Y=cell2mat(dataset.Y);

%% SVM - test training data

p = 1;     % proportion of rows to select for training
N = length(dataset.all(:,1)) ; % total number of rows 
tf = false(N,1);   % create logical index vector
tf(1:round(p*N)) = true ;    
tf = tf(randperm(N));   % randomise order

train.X = dataset.X(tf,:);
train.Y = dataset.Y(tf,:);
test.X = dataset.X(~tf,:);
test.Y = dataset.Y(~tf,:);

%% SVM vs GLM vs NaiveBayes
%dependence on samples trend the less the better

resp =train.Y(:,3);

mdl = fitglm(train.X,resp,'Distribution','binomial','Link','logit');
score_log = mdl.Fitted.Probability; % Probability estimates
[Xlog,Ylog,Tlog,AUClog] = perfcurve(resp,score_log,'true');

mdlSVM = fitcsvm(train.X,resp,'Standardize',true);
mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,mdlSVM.ClassNames),'true');

mdlNB = fitcnb(train.X,resp);
[~,score_nb] = resubPredict(mdlNB);
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,mdlNB.ClassNames),'true');

plot(Xlog,Ylog)
hold on
plot(Xsvm,Ysvm)
plot(Xnb,Ynb)
legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')
hold off 

%% 

figure
gscatter(train.X(:,11),train.X(:,10),dataset.all(:,8));
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend('Location','Northwest');

%%

SVMModels = cell(3,1);
classes = unique(dataset.all(:,8));
rng(1); % For reproducibility

for j = 1:numel(classes);
    indx = strcmp(dataset.all(:,8),classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(train.X,indx,'Standardize',true);...,...
        ... 'KernelFunction','linear');
    
    [~,score_svm] = resubPredict(mdlSVM);
    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(indx,score_svm(:,SVMModels{j}.ClassNames),'true');
    AUC{j}=AUCsvm;
    plot(Xsvm,Xsvm)
    hold on
end
legend('Art','Face','Landscape','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Classification')
hold off 

%%

