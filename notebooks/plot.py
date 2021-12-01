import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

tscv = TimeSeriesSplit(n_splits=3) 

def MAP_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_comparison(model, X, y, intervals=False, show_anomalies=False):
    
    prediction = model.predict(X)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0, c="orange")
    plt.plot(y.values, label="actual", linewidth=2.0, c="black")
    
    if intervals:
        cv = cross_val_score(model, X, y, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        # https://www.jstor.org/stable/2284575
        scale = 1.95996398454005423552
        lower, upper = prediction - (mae + scale * deviation), prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="Upper / Lower", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if show_anomalies:
            anomalies = np.array([np.NaN]*len(y))
            anomalies[y<lower] = y[y<lower]
            anomalies[y>upper] = y[y>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies", c="red")
    
    error = MAP_error(prediction, y)
    plt.title("MAP error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    