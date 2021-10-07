function label = classifyBEyebrow(X) %#codegen
% Classifies Eyebrow feature for Black face
% Reads in SVM model in file B-Eyebrow-SVM.mat then returns class label
Mdl = loadLearnerForCoder('B-Eyebrow-SVM.mat');
label = predict(Mdl,X);
end