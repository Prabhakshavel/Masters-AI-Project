function label = classifyLNose(X) %#codegen
% Classifies Nose feature for Latino face
% Reads in SVM model in file L-Nose-SVM.mat then returns class label
Mdl = loadLearnerForCoder('L-Nose-SVM.mat');
label = predict(Mdl,X);
end