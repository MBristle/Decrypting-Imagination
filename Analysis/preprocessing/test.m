SVMModels = cell(3,1);
classes = unique(dataset.all(:,8));
rng(1); % For reproducibility

for j = 1:numel(classes);
    indx = strcmp(dataset.all(:,8),classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(train.X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
    
    [~,score_svm{j}] = resubPredict(mdlSVM);
    [Xsvm{j},Ysvm{j},Tsvm,AUCsvm{j}] = perfcurve(resp,score_svm{j}(:,mdlSVM.ClassNames),'true');
end

plot(Xsvm{1},Xsvm{1})
hold on
plot(Xsvm{2},Xsvm{2})
plot(Xsvm{3},Xsvm{3})
legend('Art','Face','Landscape','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Classification')
hold off 

