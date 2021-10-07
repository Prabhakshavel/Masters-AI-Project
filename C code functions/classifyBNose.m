function label = classifyBNose(X) %#codegen
% Classifies Nose feature for Black face
% Reads in SVM model in file B-Nose-SVM.mat then returns class label
Mdl = loadLearnerForCoder('B-Nose-SVM.mat');
coder.varsize('label');
label = predict(Mdl,X);
end