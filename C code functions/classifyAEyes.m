function label = classifyAEyes(X) %#codegen
% Classifies Eyes feature for Asian face
% Reads in SVM model in file A-Eyes-SVM.mat then returns class label
Mdl = loadLearnerForCoder('A-Eyes-SVM.mat');
label = predict(Mdl,X);
end