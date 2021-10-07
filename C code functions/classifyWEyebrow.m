function label = classifyWEyebrow(X) %#codegen
% Classifies Eyebrow feature for White face
% Reads in SVM model in file W-Eyebrow-SVM.mat then returns class label
Mdl = loadLearnerForCoder('W-Eyebrow-SVM.mat');
label = predict(Mdl,X);
end