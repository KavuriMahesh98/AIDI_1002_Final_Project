# AIDI_1002_Final_Project
Original Project: NLP Text Classification Model: Disaster Tweets

The original project focuses on creating a text classification model to predict whether a given tweet is about a real disaster or not. Text classification is a fundamental task in Natural Language Processing (NLP) with numerous applications such as sentiment analysis, spam detection, topic labeling, and intent detection.

The dataset used for this project is the "Natural Language Processing with Disaster Tweets" dataset, consisting of 10,000 tweets that were hand classified. The training dataset has 7,613 labeled tweets, while the test dataset has 3,263 unlabeled tweets.

# The project follows these steps:

Setup: Importing Libraries
Loading the dataset & Exploratory Data Analysis (EDA)
Text pre-processing
Extracting vectors from text (Vectorization)
Running ML algorithms
Conclusion

# Step 2: Loading the dataset & EDA

The EDA reveals that the dataset is relatively balanced, with 57% non-disaster tweets and 43% disaster tweets. It also contains some missing values in the location and keyword columns. Disaster tweets tend to have more words and characters than non-disaster tweets.

# Step 3: Text Pre-Processing

Before building the model, the dataset is preprocessed by removing punctuations, special characters, URLs, and hashtags; correcting typos and abbreviations; removing stop words; and applying lemmatization.

# Step 4: Extracting vectors from text (Vectorization)

To work with text data in machine learning models, it needs to be converted into numerical data or vectors. This process is called vectorization or word embedding. Bag-of-Words (BoW) and Word Embedding (with Word2Vec) are two common methods for text vectorization.

In this project, the following vectorization techniques are used:

Count vectors
Term Frequency-Inverse Document Frequencies (tf-Idf)
Word2Vec
The dataset is then partitioned into an 80% training set and a 20% test set.

# Step 5: Running ML algorithms

Machine learning models are trained on the vectorized dataset and tested on the test set to evaluate performance. The following models are used:

Logistic Regression
Naive Bayes

# Limitations of Contributers work:- 
Prior approaches, as mentioned in the article, have certain limitations, such as over-reliance on the Bag-of-Words model, which ignores word order and context, leading to suboptimal performance. Additionally, these methods may not have thoroughly explored the potential of advanced word-embedding techniques, like BERT, in combination with various classification algorithms, leaving room for further investigation and improvement.

# OUR CONTRIBUTIONS

# Extracting vectors from text (Vectorization)

We additionally used BERT for text vectorization

# Evaluating using different models

we enhanced the performance of original model using the source code by implementing other classification algorithms like Support Vector Machines (SVM), XgBoost, Ensemble models, Neural networks and also used Gridsearch to tune the hyperparameters of your model

# Conclusion and Future Direction
#Comparing AUC Scores:- 

The Ensemble Model (Voting Classifier) with tf-idf has the highest AUC score (0.8557), making it the best-performing model among the classifiers tested. This indicates that combining the predictions of multiple models can lead to better overall performance.

The Support Vector Machines (SVM) Classifier with tf-idf also shows strong performance, with an AUC score of 0.8478. This demonstrates that SVM can be an effective model for this text classification problem.

The use of BERT embeddings does not significantly improve the performance when compared to the tf-idf based classifiers. This could be due to the fact that BERT might be better suited for other NLP tasks or might require further fine-tuning or optimization for this specific task.

The Logistic Regression (W2v) model has the lowest AUC score (0.6947), suggesting that Word2Vec embeddings might not be the best choice for feature extraction in this particular problem.

The results show that different classifiers have different levels of precision and recall for each class. Depending on the specific use case and requirements, one might choose a classifier that prioritizes precision or recall.

Overall, the accuracies of the different models show similar trends to their AUC scores. The Ensemble Model (Voting Classifier) with tf-idf and SVM Classifier with tf-idf are the best-performing models in terms of accuracy. However, it is essential to consider other evaluation metrics like precision, recall, and f1-score, as well as the specific requirements of the problem and the desired trade-offs between the different metrics when selecting a classifier. Further exploration of hyperparameters and feature extraction methods may also lead to improvements in model accuracy.

# Comparing Accuracies:- 

The Ensemble Model (Voting Classifier) with tf-idf has the highest accuracy score (0.80), which aligns with its highest AUC score as well. This further supports the conclusion that combining multiple models can result in improved overall performance.

The Support Vector Machines (SVM) Classifier with tf-idf and Neural Network Classifier with tf-idf also have relatively high accuracy scores of 0.80 and 0.74, respectively. This indicates that these classifiers are also effective in making correct predictions for this text classification problem.

The models using BERT embeddings show accuracies in the range of 0.69 to 0.79, which are comparable to the tf-idf based classifiers. This suggests that while BERT embeddings may not provide significant improvements over tf-idf for this task, they are still competitive in terms of accuracy.

The Logistic Regression (W2v) model has the lowest accuracy score (0.65), which is consistent with its low AUC score. This implies that the Word2Vec embeddings might not be the most suitable feature extraction method for this problem, as it leads to lower classification accuracy.

In conclusion, the Ensemble Model (Voting Classifier) with tf-idf and the SVM Classifier with tf-idf are the best-performing models for this text classification problem. However, it is important to consider the specific requirements of the task and the desired trade-offs between precision and recall when choosing a classifier. Additionally, further hyperparameter tuning and exploration of other feature extraction methods may lead to improved performance.


Through this project, we have learned that different text classification models and feature extraction techniques have varying performance, as demonstrated by the results obtained from both the author's and our contributions. The use of advanced word-embedding techniques, such as BERT, and additional classification algorithms, such as SVM, XgBoost, and Neural Networks, can lead to improved accuracies in text classification tasks.

However, the results also reveal certain limitations. For instance, some models may not perform well in certain scenarios or datasets, and there may be a trade-off between model complexity and performance. Additionally, our study focused on a specific set of feature extraction methods and classifiers, which might not cover all potential combinations.


# Future Direction:-
In future work, we could extend the analysis by incorporating other state-of-the-art embedding techniques, such as RoBERTa, GPT, or ELMo, to further explore their performance in text classification tasks. We could also evaluate the models on different datasets, representing various domains and languages, to assess their generalizability. Lastly, exploring ensemble methods, hyperparameter tuning, and other optimization techniques can potentially lead to more accurate and robust text classification models.


