function label = classifyAMouth(X) %#codegen
% Classifies Mouth feature for Asian face
% Reads in SVM model in file A-Mouth-SVM.mat then returns class label
Mdl = loadLearnerForCoder('A-Mouth-SVM.mat');
label = predict(Mdl,X);
end