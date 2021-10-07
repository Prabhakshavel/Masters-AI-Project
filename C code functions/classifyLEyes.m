function label = classifyLEyes(X) %#codegen
% Classifies Eyes feature for Latino face
% Reads in SVM model in file L-Eyes-SVM.mat then returns class label
Mdl = loadLearnerForCoder('L-Eyes-SVM.mat');
label = predict(Mdl,X);
end