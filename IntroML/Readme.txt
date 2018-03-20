https://github.com/udacity/ud120-projects.git

0. tools/startup.py 실행 - Enron data download

02. Naive Bayes     -- nb_author_id : email 내용으로 작성자 분류
03. SVM             -- svm_author_id
04. Decision Tree   -- dt_author_id

07. Regression      -- finance_regression (Enron finance data)

08. Outliers
    -- outlier_removal_regression / outlier_cleaner : residual error가 큰 10 % 버림.
    -- enron_outliers

09. Clustering -- k_means_cluster

11. Text Learning : NLTK -- SKIP!

12. Feature Selection
    -- poi_flag_email : 각 사람이 poi와 주고 받은 이메일 갯수 count
    -- poi_flag_fraction : 각 사람이 poi와 주고 받은 이메일 비율

    # tools/email_preprocess.py 참조
    from sklearn.feature_selection import SelectPercentile, SelectKBest

    SelectPercentile : selects the X% of features that are most powerful
    SelectKBest : K features that are most powerful

    high bias data, low r square, large SSE --> few features used
    high variance data --> many features, carefully optimized performance on training data


13. PCA

14. Validation

15. Evaluation Metrics





