function label = classifyWEyes(X) %#codegen
% Classifies Eyes feature for White face
% Reads in SVM model in file W-Eyes-SVM.mat then returns class label
Mdl = loadLearnerForCoder('W-Eyes-SVM.mat');
label = predict(Mdl,X);
end