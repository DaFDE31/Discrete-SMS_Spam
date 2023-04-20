import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_table('SMSSpamCollection',names=['label','sms_message'])
#df = pd.read_table('../input/sms-spam-detection/smsspamcollection/SMSSpamCollection',names=['label','sms_message'])

# Printing out first five columns
df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})

df.head()

df.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))