{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":9120,"databundleVersionId":860599,"sourceType":"competition"}],"dockerImageVersionId":11105,"isInternetEnabled":false,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\n# HOME CREDIT DEFAULT RISK COMPETITION\n# Most features are created by applying min, max, mean, sum and var functions to grouped tables. \n# Little feature selection is done and overfitting might be a problem since many features are related.\n# The following key ideas were used:\n# - Divide or subtract important features to get rates (like annuity and income)\n# - In Bureau Data: create specific features for Active credits and Closed credits\n# - In Previous Applications: create specific features for Approved and Refused applications\n# - Modularity: one function for each table (except bureau_balance and application_test)\n# - One-hot encoding for categorical features\n# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).\n# You can use LightGBM with KFold or Stratified KFold.\n\n# Update 16/06/2018:\n# - Added Payment Rate feature\n# - Removed index from features\n# - Use standard KFold CV (not stratified)\n\nimport numpy as np\nimport pandas as pd\nimport gc\nimport time\nfrom contextlib import contextmanager\nfrom lightgbm import LGBMClassifier\nfrom sklearn.metrics import roc_auc_score, roc_curve\nfrom sklearn.model_selection import KFold, StratifiedKFold\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.simplefilter(action='ignore', category=FutureWarning)\n\n@contextmanager\ndef timer(title):\n    t0 = time.time()\n    yield\n    print(\"{} - done in {:.0f}s\".format(title, time.time() - t0))\n\n# One-hot encoding for categorical columns with get_dummies\ndef one_hot_encoder(df, nan_as_category = True):\n    original_columns = list(df.columns)\n    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']\n    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)\n    new_columns = [c for c in df.columns if c not in original_columns]\n    return df, new_columns\n\n# Preprocess application_train.csv and application_test.csv\ndef application_train_test(num_rows = None, nan_as_category = False):\n    # Read data and merge\n    df = pd.read_csv('../input/application_train.csv', nrows= num_rows)\n    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)\n    print(\"Train samples: {}, test samples: {}\".format(len(df), len(test_df)))\n    df = df.append(test_df).reset_index()\n    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)\n    df = df[df['CODE_GENDER'] != 'XNA']\n    \n    # Categorical features with Binary encode (0 or 1; two categories)\n    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:\n        df[bin_feature], uniques = pd.factorize(df[bin_feature])\n    # Categorical features with One-Hot encode\n    df, cat_cols = one_hot_encoder(df, nan_as_category)\n    \n    # NaN values for DAYS_EMPLOYED: 365.243 -> nan\n    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)\n    # Some simple new features (percentages)\n    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']\n    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']\n    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']\n    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']\n    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']\n    del test_df\n    gc.collect()\n    return df\n\n# Preprocess bureau.csv and bureau_balance.csv\ndef bureau_and_balance(num_rows = None, nan_as_category = True):\n    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)\n    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)\n    bb, bb_cat = one_hot_encoder(bb, nan_as_category)\n    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)\n    \n    # Bureau balance: Perform aggregations and merge with bureau.csv\n    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}\n    for col in bb_cat:\n        bb_aggregations[col] = ['mean']\n    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)\n    bb_agg.columns = pd.Index([e[0] + \"_\" + e[1].upper() for e in bb_agg.columns.tolist()])\n    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')\n    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)\n    del bb, bb_agg\n    gc.collect()\n    \n    # Bureau and bureau_balance numeric features\n    num_aggregations = {\n        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],\n        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],\n        'DAYS_CREDIT_UPDATE': ['mean'],\n        'CREDIT_DAY_OVERDUE': ['max', 'mean'],\n        'AMT_CREDIT_MAX_OVERDUE': ['mean'],\n        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],\n        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],\n        'AMT_CREDIT_SUM_OVERDUE': ['mean'],\n        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],\n        'AMT_ANNUITY': ['max', 'mean'],\n        'CNT_CREDIT_PROLONG': ['sum'],\n        'MONTHS_BALANCE_MIN': ['min'],\n        'MONTHS_BALANCE_MAX': ['max'],\n        'MONTHS_BALANCE_SIZE': ['mean', 'sum']\n    }\n    # Bureau and bureau_balance categorical features\n    cat_aggregations = {}\n    for cat in bureau_cat: cat_aggregations[cat] = ['mean']\n    for cat in bb_cat: cat_aggregations[cat + \"_MEAN\"] = ['mean']\n    \n    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})\n    bureau_agg.columns = pd.Index(['BURO_' + e[0] + \"_\" + e[1].upper() for e in bureau_agg.columns.tolist()])\n    # Bureau: Active credits - using only numerical aggregations\n    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]\n    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)\n    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + \"_\" + e[1].upper() for e in active_agg.columns.tolist()])\n    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')\n    del active, active_agg\n    gc.collect()\n    # Bureau: Closed credits - using only numerical aggregations\n    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]\n    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)\n    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + \"_\" + e[1].upper() for e in closed_agg.columns.tolist()])\n    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')\n    del closed, closed_agg, bureau\n    gc.collect()\n    return bureau_agg\n\n# Preprocess previous_applications.csv\ndef previous_applications(num_rows = None, nan_as_category = True):\n    prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)\n    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)\n    # Days 365.243 values -> nan\n    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)\n    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)\n    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)\n    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)\n    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)\n    # Add feature: value ask / value received percentage\n    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']\n    # Previous applications numeric features\n    num_aggregations = {\n        'AMT_ANNUITY': ['min', 'max', 'mean'],\n        'AMT_APPLICATION': ['min', 'max', 'mean'],\n        'AMT_CREDIT': ['min', 'max', 'mean'],\n        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],\n        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],\n        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],\n        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],\n        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],\n        'DAYS_DECISION': ['min', 'max', 'mean'],\n        'CNT_PAYMENT': ['mean', 'sum'],\n    }\n    # Previous applications categorical features\n    cat_aggregations = {}\n    for cat in cat_cols:\n        cat_aggregations[cat] = ['mean']\n    \n    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})\n    prev_agg.columns = pd.Index(['PREV_' + e[0] + \"_\" + e[1].upper() for e in prev_agg.columns.tolist()])\n    # Previous Applications: Approved Applications - only numerical features\n    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]\n    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)\n    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + \"_\" + e[1].upper() for e in approved_agg.columns.tolist()])\n    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')\n    # Previous Applications: Refused Applications - only numerical features\n    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]\n    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)\n    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + \"_\" + e[1].upper() for e in refused_agg.columns.tolist()])\n    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')\n    del refused, refused_agg, approved, approved_agg, prev\n    gc.collect()\n    return prev_agg\n\n# Preprocess POS_CASH_balance.csv\ndef pos_cash(num_rows = None, nan_as_category = True):\n    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)\n    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)\n    # Features\n    aggregations = {\n        'MONTHS_BALANCE': ['max', 'mean', 'size'],\n        'SK_DPD': ['max', 'mean'],\n        'SK_DPD_DEF': ['max', 'mean']\n    }\n    for cat in cat_cols:\n        aggregations[cat] = ['mean']\n    \n    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)\n    pos_agg.columns = pd.Index(['POS_' + e[0] + \"_\" + e[1].upper() for e in pos_agg.columns.tolist()])\n    # Count pos cash accounts\n    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()\n    del pos\n    gc.collect()\n    return pos_agg\n    \n# Preprocess installments_payments.csv\ndef installments_payments(num_rows = None, nan_as_category = True):\n    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)\n    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)\n    # Percentage and difference paid in each installment (amount paid and installment value)\n    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']\n    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']\n    # Days past due and days before due (no negative values)\n    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']\n    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']\n    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)\n    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)\n    # Features: Perform aggregations\n    aggregations = {\n        'NUM_INSTALMENT_VERSION': ['nunique'],\n        'DPD': ['max', 'mean', 'sum'],\n        'DBD': ['max', 'mean', 'sum'],\n        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],\n        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],\n        'AMT_INSTALMENT': ['max', 'mean', 'sum'],\n        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],\n        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']\n    }\n    for cat in cat_cols:\n        aggregations[cat] = ['mean']\n    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)\n    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + \"_\" + e[1].upper() for e in ins_agg.columns.tolist()])\n    # Count installments accounts\n    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()\n    del ins\n    gc.collect()\n    return ins_agg\n\n# Preprocess credit_card_balance.csv\ndef credit_card_balance(num_rows = None, nan_as_category = True):\n    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)\n    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)\n    # General aggregations\n    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)\n    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])\n    cc_agg.columns = pd.Index(['CC_' + e[0] + \"_\" + e[1].upper() for e in cc_agg.columns.tolist()])\n    # Count credit card lines\n    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()\n    del cc\n    gc.collect()\n    return cc_agg\n\n# LightGBM GBDT with KFold or Stratified KFold\n# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code\ndef kfold_lightgbm(df, num_folds, stratified = False, debug= False):\n    # Divide in training/validation and test data\n    train_df = df[df['TARGET'].notnull()]\n    test_df = df[df['TARGET'].isnull()]\n    print(\"Starting LightGBM. Train shape: {}, test shape: {}\".format(train_df.shape, test_df.shape))\n    del df\n    gc.collect()\n    # Cross validation model\n    if stratified:\n        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)\n    else:\n        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)\n    # Create arrays and dataframes to store results\n    oof_preds = np.zeros(train_df.shape[0])\n    sub_preds = np.zeros(test_df.shape[0])\n    feature_importance_df = pd.DataFrame()\n    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]\n    \n    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):\n        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]\n        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]\n\n        # LightGBM parameters found by Bayesian optimization\n        clf = LGBMClassifier(\n            nthread=4,\n            n_estimators=10000,\n            learning_rate=0.02,\n            num_leaves=34,\n            colsample_bytree=0.9497036,\n            subsample=0.8715623,\n            max_depth=8,\n            reg_alpha=0.041545473,\n            reg_lambda=0.0735294,\n            min_split_gain=0.0222415,\n            min_child_weight=39.3259775,\n            silent=-1,\n            verbose=-1, )\n\n        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], \n            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)\n\n        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]\n        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits\n\n        fold_importance_df = pd.DataFrame()\n        fold_importance_df[\"feature\"] = feats\n        fold_importance_df[\"importance\"] = clf.feature_importances_\n        fold_importance_df[\"fold\"] = n_fold + 1\n        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))\n        del clf, train_x, train_y, valid_x, valid_y\n        gc.collect()\n\n    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))\n    # Write submission file and plot feature importance\n    if not debug:\n        test_df['TARGET'] = sub_preds\n        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)\n    display_importances(feature_importance_df)\n    return feature_importance_df\n\n# Display/plot feature importance\ndef display_importances(feature_importance_df_):\n    cols = feature_importance_df_[[\"feature\", \"importance\"]].groupby(\"feature\").mean().sort_values(by=\"importance\", ascending=False)[:40].index\n    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]\n    plt.figure(figsize=(8, 10))\n    sns.barplot(x=\"importance\", y=\"feature\", data=best_features.sort_values(by=\"importance\", ascending=False))\n    plt.title('LightGBM Features (avg over folds)')\n    plt.tight_layout()\n    plt.savefig('lgbm_importances01.png')\n\n\ndef main(debug = False):\n    num_rows = 10000 if debug else None\n    df = application_train_test(num_rows)\n    with timer(\"Process bureau and bureau_balance\"):\n        bureau = bureau_and_balance(num_rows)\n        print(\"Bureau df shape:\", bureau.shape)\n        df = df.join(bureau, how='left', on='SK_ID_CURR')\n        del bureau\n        gc.collect()\n    with timer(\"Process previous_applications\"):\n        prev = previous_applications(num_rows)\n        print(\"Previous applications df shape:\", prev.shape)\n        df = df.join(prev, how='left', on='SK_ID_CURR')\n        del prev\n        gc.collect()\n    with timer(\"Process POS-CASH balance\"):\n        pos = pos_cash(num_rows)\n        print(\"Pos-cash balance df shape:\", pos.shape)\n        df = df.join(pos, how='left', on='SK_ID_CURR')\n        del pos\n        gc.collect()\n    with timer(\"Process installments payments\"):\n        ins = installments_payments(num_rows)\n        print(\"Installments payments df shape:\", ins.shape)\n        df = df.join(ins, how='left', on='SK_ID_CURR')\n        del ins\n        gc.collect()\n    with timer(\"Process credit card balance\"):\n        cc = credit_card_balance(num_rows)\n        print(\"Credit card balance df shape:\", cc.shape)\n        df = df.join(cc, how='left', on='SK_ID_CURR')\n        del cc\n        gc.collect()\n    with timer(\"Run LightGBM with kfold\"):\n        feat_importance = kfold_lightgbm(df, num_folds= 10, stratified= False, debug= debug)\n\nif __name__ == \"__main__\":\n    submission_file_name = \"submission_kernel02.csv\"\n    with timer(\"Full model run\"):\n        main()","metadata":{"_uuid":"3ad1fdf1-c793-497d-8ed1-66771bcf01ef","_cell_guid":"32b77957-9af5-4297-b79f-ac4a38b44cec","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}