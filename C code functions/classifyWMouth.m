function label = classifyWMouth(X) %#codegen
% Classifies Mouth feature for White face
% Reads in SVM model in file W-Mouth-SVM.mat then returns class label
Mdl = loadLearnerForCoder('W-Mouth-SVM.mat');
label = predict(Mdl,X);
end