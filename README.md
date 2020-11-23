# Project
This project is one of the deliverables in the couse TDT4173 - Machine Learning at NTNU conducted by a group of three students. 
\
The repository contains a **CNN model built to classify real and artificial images of human faces**. 

The purpose of this project is the academic learning of adressing a machine learning task.

### Running the application 
The application can be run with a Virtual Machine, the following setup is with Google Colab: 

First you have to upload the dataset from kaggle on Google Drive:
https://www.kaggle.com/ciplab/real-and-fake-face-detection

**Fetch the dataset from Google Colab**

1. Executing the below code which will provide you with an authentication link

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

2. Open the link

3. Choose the Google account whose Drive you want to mount

4. Allow Google Drive Stream access to your Google Account

5. Copy the code displayed, paste it in the text box as shown below, and press Enter colab import drive

Once the Drive is mounted, you’ll get the message **“Mounted at /content/drive”**, and you’ll be able to browse through the contents of your Drive from the file-explorer pane. 

Now you can interact with your Google Drive as if it was a folder.
Exampel from the model:
`fake = "/content/drive/My Drive/archive/real_and_fake_face/training_fake"`

**The setup is complete and you can run the code**
