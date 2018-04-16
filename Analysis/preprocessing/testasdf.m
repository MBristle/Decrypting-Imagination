SVMModels = cell(3,1);
classes = unique(dataset.all(:,8));
rng(1); % For reproducibility

for j = 1:numel(classes);
    indx = strcmp(dataset.all(:,8),classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(train.X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
    
    [~,score_svm] = resubPredict(mdlSVM);
    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,mdlSVM.ClassNames),'true');
    AUC{j}=AUCsvm;
    plot(Xsvm,Xsvm)
    hold on
end
legend('Art','Face','Landscape','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Classification')
hold off 

