# Project
This project is one of the deliverables in the couse TDT4173 - Machine Learning at NTNU. 

The purpose of this project is the academic learning of adressing a machine learning task.

## Running the application 
You can run and test the code with a Virtual Machine or iPhyton notebook. 

To run the application you have to download the dataset from kaggle:
https://www.kaggle.com/ciplab/real-and-fake-face-detection

### Fetch the dataset

When saving the dataset on google drive you can access it in Google Colab with the following code: 

1. Executing the below code which will provide you with an authentication link

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

2. Open the link

3. Choose the Google account whose Drive you want to mount

4. Allow Google Drive Stream access to your Google Account

5. Copy the code displayed, paste it in the text box as shown below, and press Enter

colab import drive
Once the Drive is mounted, you’ll get the message “Mounted at /content/drive”, and you’ll be able to browse through the contents of your Drive from the file-explorer pane.

You can then access the dataset as usual: 
accessing = "write the path to de folder here"

Exampel from the model:
`fake = "/content/drive/My Drive/archive/real_and_fake_face/training_fake"`


Tidligere prosjekt: 
https://github.com/Mael7307/Fake-image-detector-CNN-
