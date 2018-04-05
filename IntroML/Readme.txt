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


11. Text Learning : NLTK

    --------------------------------------------------------------------------
    conda install nltk
    import nltk
    nltk.download()  --> 링크 클릭 --> GUI에서 all-copora 다운로드
    sw = stopwords.words("english")  # 에러 없으면 설치 성공
    --------------------------------------------------------------------------

    # stemming --> bag of word --> TfIdf vectorization
    # TfIdf : Tf - term frequency. Idf - inverse document frequency. (inverse = Upweight rare words)

    tools/parse_out_email_text : nltk 사용해 stemming words 추출
    vectorize_text : parse_out_email_text를 통해 추출한 두 사람의 이메일 단어를 TfIdf vecterization


12. Feature Selection
    -- poi_flag_email : 각 사람이 poi와 주고 받은 이메일 갯수 count
    -- poi_flag_fraction : 각 사람이 poi와 주고 받은 이메일 비율
    -- find_signature : overfitting 유발하는 outlier check

    --------------------------------------------------------------------------
    # tools/email_preprocess.py 참조
    from sklearn.feature_selection import SelectPercentile, SelectKBest

    SelectPercentile : selects the X% of features that are most powerful
    SelectKBest : K features that are most powerful

    high bias data, low r square, large SSE --> few features used
    high variance data --> many features, carefully optimized performance on training data
    --------------------------------------------------------------------------


13. PCA
    -- eigenfaces (http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)
    PCA --> SVM --> classify faces


14. Validation : GridSearchCV

    sklearn.model_selection.GridSearchCV
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters) --> (linear, 1) / (linear, 10) / (rbf, 1) / (rbf, 10)
