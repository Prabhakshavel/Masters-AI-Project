function label = classifyBMouth(X) %#codegen
% Classifies Mouth feature for Black face
% Reads in SVM model in file B-Mouth-SVM.mat then returns class label
Mdl = loadLearnerForCoder('B-Mouth-SVM.mat');
coder.varsize('X');
label = predict(Mdl,X);
end