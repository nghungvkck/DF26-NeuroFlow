from abc import ABC


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class cho các mô hình dự báo chuỗi thời gian.
    """

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Huấn luyện mô hình.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix của tập train
        y_train : pd.Series or np.ndarray
            Target của tập train
        X_valid : pd.DataFrame, optional
            Feature matrix của tập validation
        y_valid : pd.Series or np.ndarray, optional
            Target của tập validation
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Dự báo trên dữ liệu mới.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        np.ndarray
            Giá trị dự báo
        """
        raise NotImplementedError

    def save(self, path):
        """
        Lưu mô hình ra file.

        Parameters
        ----------
        path : str
            Đường dẫn file lưu model
        """
        raise NotImplementedError

    def load(self, path):
        """
        Load mô hình từ file.

        Parameters
        ----------
        path : str
            Đường dẫn file model
        """
        raise NotImplementedError
