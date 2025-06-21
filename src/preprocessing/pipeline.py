from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
import numpy as np


ORDINAL = ['Status', 'CreditHistory', 'Savings', 'EmploymentSince']
CATEGORICAL_NOMINAL = ['Purpose', 'PersonalStatusSex', 'Debtors', 'Property', 'OtherInstallmentPlans', 'Housing', 'Job', 'Telephone', 'ForeignWorker']
CONTINUOUS = ['CreditAmount']

# Function Log-transforming Skewed Numerical Feature
def log_transform(X):
    return np.log1p(X)

# Building the Continuous Value Feature Transformer
continuous_transformer = Pipeline([
    ('log', FunctionTransformer(log_transform, validate=False))
])

# Building the Categorical Nominal Feature Transformer
nominal_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Building the Ordinal Feature Mapper
ordinal_mapping = {
    'Status': ['A11', 'A12', 'A13', 'A14'],
    'CreditHistory': ['A30', 'A31', 'A32', 'A33', 'A34'],
    'Savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
    'EmploymentSince': ['A71', 'A72', 'A73', 'A74', 'A75']
}
ordinal_transformer = OrdinalEncoder(
    categories=[ordinal_mapping[col] for col in ORDINAL],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

preprocessor = ColumnTransformer([
    ('continuous', continuous_transformer, CONTINUOUS),
    ('nominal', nominal_transformer, CATEGORICAL_NOMINAL),
    ('ordinal', ordinal_transformer, ORDINAL)
])

def build_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])