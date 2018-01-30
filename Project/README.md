############################################################################################################################
################################               FACIAL EXPRESSION PREDICTION            #####################################   
############################################################################################################################

Follow below steps to execute the code.

Install all python modules mentioned in requirements.txt by using below command.

pip install -r requirements.txt

Copy CK+ dataset, data into data/original_data folder and labels into data/labels folder.

Execute Data_Organiser_Augmentor.ipynb to get filtered, splitted, alligned and augmented dataset 
as per the requirement to run the model smoothly. 
After this, you will have well separated aligned dataset in data/set folder under respective emotion folder.

Execute Naive_Expnet.ipynb to train the model.
We are saving the model in model/ folder.
It generally takes max upto 1 hour to complete under normal configuration that has been set inside the file.

Once you are good with the model. Upload some images into data/predict_images folder.
Execute Prediction.ipynb with your trained model to predict the expression of recently uploaded rel time expressive pics of yours.

Cheers....

Jyotirmay Senapati
Shayan Ahmad Siddiqui
Abhijeet Parida
DL4CV Winter Sem Project.
Technical University of Munich, Germany.

