function label = classifyWNose(X) %#codegen
% Classifies Nose feature for White face
% Reads in SVM model in file W-Nose-SVM.mat then returns class label
Mdl = loadLearnerForCoder('W-Nose-SVM.mat');
label = predict(Mdl,X);
end