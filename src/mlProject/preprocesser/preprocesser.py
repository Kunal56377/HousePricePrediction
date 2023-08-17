from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin

class Preprocessor(BaseEstimator,TransformerMixin):
    # Train our custom preprocessors
    def fit(self,X,y=None):
        self.imputer = SimpleImputer()
        self.imputer.fit(X[['HouseAge','DistanceToStation','NumberOfPubs']])

        self.scaler = StandardScaler()
        self.scaler.fit(X[['HouseAge','DistanceToStation','NumberOfPubs']])

        self.onehot = OneHotEncoder(handle_unknown = 'ignore')
        X['PostCode'] = X.PostCode.astype(str)
        self.onehot.fit(X[['PostCode']])

        return self

    
    # Apply our custom preprocessors 
    def transform(self,X):
        #Apply Simple imputer
       
        X['TransactionDate'] = X.TransactionDate.astype(str)
        X['PostCode'] = X.PostCode.astype(str)

        imputed_cols=self.imputer.transform(X[['HouseAge','DistanceToStation','NumberOfPubs']])
        onehot_cols = self.onehot.transform(X[['PostCode']])

        # Copy the df
        transformed_df = X.copy()
        transformed_df['Year'] = transformed_df['TransactionDate'].apply(lambda X:X[:4]).astype(int)
        transformed_df['Month'] = transformed_df['TransactionDate'].apply(lambda X:X[5:]).astype(int)
        transformed_df = transformed_df.drop('TransactionDate',axis=1)
        
        # Apply transformed columns 
        transformed_df[['HouseAge','DistanceToStation','NumberOfPubs']] = imputed_cols
        transformed_df[['HouseAge','DistanceToStation','NumberOfPubs']] =  self.scaler.transform(transformed_df[['HouseAge','DistanceToStation','NumberOfPubs']])
        transformed_df = transformed_df.drop('PostCode',axis=1)
        transformed_df[self.onehot.get_feature_names_out()] = onehot_cols.toarray().astype(int)
        # print(transformed_df.shape)
        return transformed_df