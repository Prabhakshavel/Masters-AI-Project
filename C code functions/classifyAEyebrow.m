function label = classifyAEyebrow(X) %#codegen
% Classifies Eyebrow feature for Asian face
% Reads in SVM model in file A-Eyebrow-SVM.mat then returns class label
Mdl = loadLearnerForCoder('A-Eyebrow-SVM.mat');
label = predict(Mdl,X);
end