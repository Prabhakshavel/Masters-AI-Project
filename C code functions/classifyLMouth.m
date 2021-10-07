function label = classifyLMouth(X) %#codegen
% Classifies Mouth feature for Latino face
% Reads in SVM model in file L-Mouth-SVM.mat then returns class label
Mdl = loadLearnerForCoder('L-Mouth-SVM.mat');
label = predict(Mdl,X);
end