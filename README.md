
# Application of machine-learning computer vision to videogame character customisation

Project created for Master's project at Loughborough University.

This project was produced in Python and Matlab, with minor C usage.

The project was an investigation into the use of machine learning to automate character creation in videogames. In the completed workflow, a user would upload an image of the target face to the preprocessing application. This would isolate and extract the features of the target, for use later. The classification Matlab functionality could then be used to identify the class of each feature, which would be mapped to a specific feature in each game engine. 

In this implementation, Monster Hunter: World and the Nintendo Mii creator were chosen as character creator engines. The design of this classifier is such that it is capable of being expanded to any similar creator with ease, simply requiring a new mapping to be produced for the new creator.  

# Celebrity face appromixation

### Base image for generation
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/L2.jpg" height="550">
<br/><br/>

### Trimmed for image processing
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/Trimmed/L2.jpg" height="550">
<br/><br/>

### Processed feature example: Nose
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/Features/Noses/L2.png" height="550">
<br/><br/>

### Processed feature example: Eye
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/Features/Eyes/L2-1.png" height="250">
<br/><br/>

### Processed feature example: Mouth
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/Features/Mouths/L2.png" height="250">
<br/><br/>

### Generated Avatar example: Monster Hunter:World
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/Avatars/MHW/L2.png" height="550">
<br/><br/>

### Generated Avatar example: Nintendo Mii creator
<br/><br/>
<img src="https://raw.githubusercontent.com/Prabhakshavel/Masters-AI-Project/main/Celebrity%20samples/Avatars/Mii/L2.PNG" height="550">
<br/><br/>

