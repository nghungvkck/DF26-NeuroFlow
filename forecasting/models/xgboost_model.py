import xgboost as xgb
from forecasting.models.base_model import BaseTimeSeriesModel


class XGBoostModel(BaseTimeSeriesModel):
    """
    XGBoost regression model cho bài toán dự báo chuỗi thời gian
    """

    name = "xgboost"

    def __init__(self, params, num_boost_round=1000, early_stopping=50):
        """
        Parameters
        ----------
        params : dict
            Tham số huấn luyện của XGBoost
        num_boost_round : int, default=1000
            Số vòng boosting tối đa
        early_stopping : int, default=50
            Số vòng early stopping
        """
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Huấn luyện mô hình XGBoost.

        Nếu có tập validation thì sử dụng early stopping.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix train
        y_train : pd.Series
            Target train
        X_valid : pd.DataFrame, optional
            Feature matrix validation
        y_valid : pd.Series, optional
            Target validation
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = []
        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            evals = [(dvalid, "valid")]

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping if evals else None,
            verbose_eval=False
        )

    def predict(self, X):
        """
        Dự báo bằng mô hình đã huấn luyện.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        np.ndarray
            Giá trị dự báo
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path):
        """
        Lưu model XGBoost ra file.

        Parameters
        ----------
        path : str
        """
        self.model.save_model(path)

    @classmethod
    def load(cls, path):
        """
        Load model XGBoost từ file.

        Parameters
        ----------
        path : str
            Đường dẫn file model

        Returns
        -------
        XGBoostModel
            Instance của XGBoostModel
        """
        obj = cls(params={})
        obj.model = xgb.Booster()
        obj.model.load_model(path)
        return obj
