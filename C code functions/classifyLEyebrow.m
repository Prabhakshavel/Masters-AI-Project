function label = classifyLEyebrow(X) %#codegen
% Classifies Eyebrow feature for Latino face
% Reads in SVM model in file L-Eyebrow-SVM.mat then returns class label
Mdl = loadLearnerForCoder('L-Eyebrow-SVM.mat');
label = predict(Mdl,X);
end