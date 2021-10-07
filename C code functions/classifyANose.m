function label = classifyANose(X) %#codegen
% Classifies Nose feature for Asian face
% Reads in SVM model in file A-Nose-SVM.mat then returns class label
Mdl = loadLearnerForCoder('A-Nose-SVM.mat');
label = predict(Mdl,X);
end