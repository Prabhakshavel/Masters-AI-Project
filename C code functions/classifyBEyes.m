function label = classifyBEyes(X) %#codegen
% Classifies Eyes feature for Black face
% Reads in SVM model in file B-Eyes-SVM.mat then returns class label
Mdl = loadLearnerForCoder('B-Eyes-SVM.mat');
label = predict(Mdl,X);
end