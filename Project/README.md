

## FACIAL EXPRESSION PREDICTION       

* The application is running here: [https://iexp.jyotirmays.com](https://iexp.jyotirmays.com/)

Follow below steps to execute the code.<br/>

You need to have python 3.6 with anaconda installed on your machine. <br/>
Install all python modules mentioned in requirements.txt by using below command.

  > pip install -r requirements.txt

Copy CK+ dataset, `data` into `data/original_data` folder and `labels` into `data/labels` folder.

Execute **_`Data_Organiser_Augmentor.ipynb`_** to get filtered, splitted, alligned and augmented dataset 
as per the requirement to run the model smoothly.
After this, you will have well separated `aligned dataset` in `data/set` folder under respective emotion folder.

Execute **_`Naive_Expnet.ipynb`_** to train the model.
We are saving the `model` in `model/` folder.
It generally takes max upto 1 hour to complete under normal configuration that has been set inside the file.

Once you are good with the model. Upload some images into `data/predict_images` folder.<br/>
Execute **_`Prediction.ipynb`_** with your trained model to predict the expression of recently uploaded real time expressive pics of yours.

You can skip training a new model and run prediction with our pre-trained model stored in `model` folder.<br/>

If you do not have a GPU supported system, you can try executing **_`Prediction_CPU.ipynb`_** file on CPU machine.

Cheers....

- [Jyotirmay Senapati](https://www.linkedin.com/in/jyotirmay-senapati-30615421/)
- [Shayan Siddiqui](https://www.linkedin.com/in/shayan-siddiqui/)
- [Abhijeet Parida](https://www.linkedin.com/in/a-parida/)

**DL4CV Winter Sem Project, 2018.**
<br/>
**Technical University of Munich, Germany.**


