import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


TARGET = "satisfaction"

CATEGORICAL_BINARY = ["Gender", "Customer Type", "Type of Travel"]
CATEGORICAL_MULTI  = ["Class"]
NUMERICAL = [
    "Age",
    "Flight Distance",
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")),
    ])
    multi_pipe = Pipeline([
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe,   NUMERICAL),
        ("bin", bin_pipe,   CATEGORICAL_BINARY),
        ("cat", multi_pipe, CATEGORICAL_MULTI),
    ])


def prepare(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[TARGET]), index=df.index, name=TARGET)
    X = df[CATEGORICAL_BINARY + CATEGORICAL_MULTI + NUMERICAL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, le
