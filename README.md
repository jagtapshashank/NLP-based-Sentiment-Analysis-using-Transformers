# NLP-based-Sentiment-Analysis-using-Transformers
Developed this project to classify restaurant reviews and analyze customer sentiment. Analyzing customer sentiment in restaurant reviews provides valuable insights into customer satisfaction and areas for improvement. It helps businesses identify trends, address negative feedback, and enhance the overall dining experience. Automated sentiment analysis enables faster processing of large volumes of reviews, offering timely insights to restaurant managers. This can improve decision-making, boost customer engagement, and ultimately drive better business outcomes through more responsive service.

## Model Development
The model development began by importing the training and testing datasets in CSV format, each containing two columns: 'text' (reviews) and 'stars' (ratings). During preprocessing, text was converted to lowercase, punctuation was removed, and stopwords were filtered using the NLTK library. A new ‘sentiment’ column was introduced, categorizing ratings as positive (above 3), neutral (3), or negative (≤2). The text was tokenized, converted into sequences of integer indices, and padded to ensure uniform sequence lengths for input to the machine learning model. For model implementation, BERT (bert-base-cased) was employed using PyTorch, freezing the first three transformer layers for efficiency. A custom classifier was built with linear layers, ReLU activations, and a dropout rate of 40%. The model was trained over five epochs using stochastic gradient descent (SGD) with a learning rate of 0.01 and momentum of 0.9, saving the best model based on validation accuracy.

## Results
The model achieved 82% testing accuracy, effectively classifying most reviews. It performed well on 'Positive' and 'Neutral' sentiments but struggled with 'Negative' predictions, suggesting an imbalance in the label distribution. The classification report and confusion matrix highlighted lower F1-scores for 'Negative' sentiment, indicating fewer instances of negative reviews in the training data. Despite signs of overfitting after the 4th epoch, the final model maintained robust performance, balancing accuracy and generalization across unseen data.

<img width="549" alt="classification_report" src="https://github.com/user-attachments/assets/f9d336d6-e5da-42b5-b447-25f99dc3a8bf">
<img width="549" alt="confusion_matrix" src="https://github.com/user-attachments/assets/8d52089b-30c2-440e-b64c-67b05366a68a">

## Conclusion
In conclusion, we implemented a Transformer-based sentiment analysis model using BERT on a subset of the Yelp review dataset. Careful data preprocessing improved model performance, and the model, trained over five epochs with SGD, achieved 85.84% validation accuracy and 82% testing accuracy. However, class imbalance impacted the prediction of 'Negative' sentiments. This work highlights the importance of balanced datasets and showcases the effectiveness of BERT in capturing contextual information for sentiment analysis. Overall, the project offers insights into model development and demonstrates the potential of Transformer architectures in NLP tasks.

## How to run 
Download yelp reviews train and test datasets from here: https://drive.google.com/drive/folders/1JsoRb_6hqWbF3S4pSe1kvSHKAaNq43nN?usp=sharing

After that run ShashankJagtap_MLproject.ipynb step by step.

## Caution: This project demands significant computational power for model training and should only be run on systems equipped with a powerful GPU.
