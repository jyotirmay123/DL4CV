

## FACIAL EXPRESSION PREDICTION       


Follow below steps to execute the code.<br/>

Install all python modules mentioned in requirements.txt by using below command.<br/>

> pip install -r requirements.txt

Copy CK+ dataset, `data` into `data/original_data` folder and `labels` into `data/labels` folder.<br/>

Execute **_`Data_Organiser_Augmentor.ipynb`_** to get filtered, splitted, alligned and augmented dataset 
as per the requirement to run the model smoothly. <br/>
After this, you will have well separated `aligned dataset` in `data/set` folder under respective emotion folder.<br/>

Execute **_`Naive_Expnet.ipynb`_** to train the model.<br/>
We are saving the `model` in `model/` folder.<br/>
It generally takes max upto 1 hour to complete under normal configuration that has been set inside the file.<br/>

Once you are good with the model. Upload some images into `data/predict_images` folder.<br/>
Execute **_`Prediction.ipynb`_** with your trained model to predict the expression of recently uploaded real time expressive pics of yours.<br/>

If you do not have a GPU supported system, you can try executing **_`Prediction_CPU.ipynb`_** file on CPU machine.<br/>

Cheers....

- **Jyotirmay Senapati**
- **Shayan Ahmad Siddiqui**
- **Abhijeet Parida**

**DL4CV Winter Sem Project, 2018.**
<br/>
**Technical University of Munich, Germany.**

