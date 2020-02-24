from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import pandas as pd
class_label = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
def prepare_df(path, train=False, nrows=None):
    """
    Prepare Pandas DataFrame for fitting neural network models
    Returns a Dataframe with two columns
    ImageID and Labels (list of all labels for an image)
    """
    df = pd.read_csv(path, nrows=nrows)
    if train:
        # Duplicates found from this kernel:
        # https://www.kaggle.com/akensert/resnet50-keras-baseline-model
        duplicates_to_remove = [1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
                                312468,  312469,  312470,  312471,  312472,  312473,
                                2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
                                3032994, 3032995, 3032996, 3032997, 3032998, 3032999]
        df = df.drop(index=duplicates_to_remove).reset_index(drop=True)

    # Get ImageID for using with generator
    df['ImageID'] = df['ID'].str.rsplit('_', 1).map(lambda x: x[0])
    # Get labels for each image
    label_lists = df.groupby('ImageID')['Label'].apply(list)

    # A clean DataFrame with a column for ImageID and columns for each label
    new_df = pd.DataFrame({'ImageID': df['ImageID'].unique(),
                           'Labels': label_lists}).set_index('ImageID').reset_index()
    new_df[class_label] = pd.DataFrame(new_df['Labels'].values.tolist(), index= new_df.index)
    new_df = new_df.drop('Labels', axis=1)
    return new_df
def train_test_split(df, target, classes):
    X = []
    Y = []
    for index in range(len(df)):
      image_data = df.loc[index]
      X.append(image_data[target])
      Y.append(image_data[classes].values.tolist())
    X = np.array(X)
    Y = np.array(Y)
    mskf = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.15, random_state=0)
    for train_index, test_index in mskf.split(X, Y):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = Y[train_index], Y[test_index]
    return X_train, y_train, X_test, y_test
def _normalize(img):
    return (img - img.min())/(img.max() - img.min())
