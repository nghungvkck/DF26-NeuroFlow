import lightgbm as lgb
from forecasting.models.base_model import BaseTimeSeriesModel


class LightGBMModel(BaseTimeSeriesModel):
    """
    LightGBM regression model cho bài toán dự báo chuỗi thời gian
    """

    name = "lightgbm"

    def __init__(self, params, num_boost_round=1000, early_stopping=50):
        """
        Parameters
        ----------
        params : dict
            Tham số huấn luyện của LightGBM
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
        Huấn luyện mô hình LightGBM.

        Nếu có tập validation thì dùng early stopping.

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
        train_set = lgb.Dataset(X_train, y_train)

        valid_sets = []
        if X_valid is not None and y_valid is not None:
            valid_sets.append(
                lgb.Dataset(X_valid, y_valid, reference=train_set)
            )

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(self.early_stopping),
                lgb.log_evaluation(period=0)
            ] if valid_sets else None
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
        return self.model.predict(X)

    def save(self, path):
        """
        Lưu model LightGBM ra file.

        Parameters
        ----------
        path : str
            Đường dẫn file .txt
        """
        self.model.save_model(path)

    @classmethod
    def load(cls, path):
        """
        Load model LightGBM từ file.

        Parameters
        ----------
        path : str
            Đường dẫn file model

        Returns
        -------
        LightGBMModel
            Instance của LightGBMModel
        """
        obj = cls(params={})
        obj.model = lgb.Booster(model_file=path)
        return obj
