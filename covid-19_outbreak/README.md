# Detection of COVID-19 epidemic outbreak using machine learning

This is the code used in 

'[Detection of COVID-19 epidemic outbreak using machine learning](https://www.frontiersin.org/articles/10.3389/fpubh.2023.1252357/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Public_Health&id=1252357)' published by Frontiers in Public Health.

Giphil Cho, Jeong Rye Park, Yongin Choi, Hyeonjeong Ahn, and Hyojung Lee


### Abstract

**Background:** The coronavirus disease (COVID-19) pandemic has spread rapidly across the world, creating an urgent need for predictive models that can help healthcare providers prepare and respond to outbreaks more quickly and effectively, and ultimately improve patient care. Early detection and warning systems are crucial for preventing and controlling epidemic spread.

**Objective:** In this study, we aimed to propose a machine learning-based method to predict the transmission trend of COVID-19 and a new approach to detect the start time of new outbreaks by analyzing epidemiological data.

**Methods:** We developed a risk index to measure the change in the transmission trend. We applied machine learning (ML) techniques to predict COVID-19 transmission trends, categorized into three labels: decrease (L0), maintain (L1), and increase (L2). We used support vector machine (SVM), random forest (RF), and XGBoost (XGB) as ML models. We employed grid search methods to determine the optimal hyperparameters for these three models. We proposed a new method to detect the start time of new outbreaks based on label 2, which was sustained for at least 14 days (i.e., the duration of maintenance). We compared the performance of different ML models to identify the most accurate approach for outbreak detection. We conducted sensitivity analysis for the duration of maintenance between 7  days and 28  days.

**Results:** ML methods demonstrated high accuracy (over 94%) in estimating the classification of the transmission trends. Our proposed method successfully predicted the start time of new outbreaks, enabling us to detect a total of seven estimated outbreaks, while there were five reported outbreaks between March 2020 and October 2022 in Korea. It means that our method could detect minor outbreaks. Among the ML models, the RF and XGB classifiers exhibited the highest accuracy in outbreak detection.

**Conclusion:** The study highlights the strength of our method in accurately predicting the timing of an outbreak using an interpretable and explainable approach. It could provide a standard for predicting the start time of new outbreaks and detecting future transmission trends. This method can contribute to the development of targeted prevention and control measures and enhance resource management during the pandemic.

