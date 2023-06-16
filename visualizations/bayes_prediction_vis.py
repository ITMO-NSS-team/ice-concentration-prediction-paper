import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_real_images(dates_range):
    """
    Function for forming massive of real data images with weekly time resolution
    """
    years = dates_range.dt.year.unique()
    real_data = []
    dates = []
    for year in years:
        dates_to_predict = pd.date_range(f'{year}0101', f'{year}1231', freq='D')
        dates_to_predict = [d.strftime('%Y%m%d') for d in dates_to_predict]
        dates_to_predict = dates_to_predict[::7]
        dates_to_predict = dates_to_predict[:52]
        for date in dates_to_predict:
            try:
                matrix = np.load(f'../matrices/osisaf/train/osi_iceconc_{date}.npy')
            except Exception:
                matrix = np.load(f'../matrices/osisaf/test/osi_iceconc_{date}.npy')
            real_data.append(matrix)
        dates.extend(dates_to_predict)
    real_data = np.array(real_data)
    return real_data, dates


def get_bayes_prediction(dates_range):
    """
    Function for forming massive of predictions by Bayesian networks
    """
    file_bayes = np.load('../additional_predictors/predict_bn.npy')
    dates_range = dates_range[::7]
    prediction = []
    for w in range(len(dates_range)):
        matrix = file_bayes[w, :, :]
        prediction.append(matrix)
    prediction = np.array(prediction)
    dates_range = [d.strftime('%Y%m%d') for d in dates_range]
    prediction = prediction[:52]
    dates_range = dates_range[:52]
    return prediction, dates_range


def return_mae_for_massive(massive, real_massive):
    return np.mean(abs(massive - real_massive), axis=(1, 2))


def mae(prediction, real):
    return round(float(np.mean(abs(np.array(prediction) - np.array(real)))), 3)


def imshow_real_bayes():
    start_time = '20150101'
    end_time = '20151231'
    prediction_dates = pd.Series(pd.date_range(start=start_time, end=end_time, freq='D'))

    bayes_preds, _ = get_bayes_prediction(prediction_dates)
    real, dates = get_real_images(prediction_dates)

    for i in range(real.shape[0]):
        plt.rcParams["figure.figsize"] = (10, 4)
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(real[i], vmin=0, vmax=1)
        axarr[0].title.set_text('Real data')
        axarr[1].imshow(bayes_preds[i], vmin=0, vmax=1)
        axarr[1].title.set_text(f'Bayessian model\nMAE={mae(bayes_preds[i], real[i])}')
        axarr[2].imshow(bayes_preds[i] - real[i], vmin=0, vmax=1)
        axarr[2].title.set_text(f'Bayessian model and\n real data difference distribution')

        for j in range(3):
            axarr[j].axis('off')
        f.suptitle(f'Prediction date - {dates[i]}')

        plt.savefig(f'bayes_preds/week_{i}.png', dpi=300)
        plt.show()


def show_real_bayes_by_points(dates, bs_preds, real):
    indeces = [
        [70, 55],
        [100, 80],
        [90, 105]
    ]
    matrix = np.load(f'../matrices/osisaf/train/osi_iceconc_20060101.npy')
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic([['Left', 'p1'],
                              ['Left', 'p2'],
                              ['Left', 'p3']],
                             gridspec_kw={'width_ratios': [2, 3]})
    axs['Left'].set_title(f'OSISAF ice concentration\ndate - 2013/01/01')
    for point in indeces:
        axs['Left'].scatter(*point, c='r')
        axs['Left'].text(point[0] + .03, point[1] + .03, f'{point}', fontsize=9, c='r')
    axs['Left'].imshow(matrix)

    axs['p1'].set_title(f'Point {indeces[0]}', fontsize=9)
    axs['p1'].plot(dates, bs_preds[:, indeces[0][1], indeces[0][0]],  linewidth=1, label='BN prediction')
    axs['p1'].plot(dates, real[:, indeces[0][1], indeces[0][0]], linewidth=1, label='Real data')

    axs['p1'].xaxis.set_tick_params(labelsize=7, rotation=20)
    axs['p1'].yaxis.set_tick_params(labelsize=7)

    axs['p2'].set_title(f'Point {indeces[1]}', fontsize=9)
    axs['p2'].plot(dates, bs_preds[:, indeces[1][1], indeces[1][0]],  linewidth=1, label='BN prediction')
    axs['p2'].plot(dates, real[:, indeces[1][1], indeces[1][0]],  linewidth=1, label='Real data')
    axs['p2'].xaxis.set_tick_params(labelsize=7, rotation=20)
    axs['p2'].yaxis.set_tick_params(labelsize=7)

    axs['p3'].set_title(f'Point {indeces[2]}', fontsize=9)
    axs['p3'].plot(dates, bs_preds[:, indeces[2][1], indeces[2][0]],  linewidth=1, label='BN prediction')
    axs['p3'].plot(dates, real[:, indeces[2][1], indeces[2][0]],  linewidth=1, label='Real data')
    axs['p3'].xaxis.set_tick_params(labelsize=7, rotation=20)
    axs['p3'].yaxis.set_tick_params(labelsize=7)
    plt.savefig(f'bn_ts.png', dpi=300)
    plt.show()


def plot_all_bayes_prediction():
    years = range(2013, 2018)
    all_bayes = []
    all_real = []
    all_dates = []
    for year in years:
        prediction_dates = pd.Series(pd.date_range(start=f'{year}0101', end=f'{year}1231', freq='D'))
        bayes_preds, dates = get_bayes_prediction(prediction_dates)
        real, _ = get_real_images(prediction_dates)
        all_bayes.extend(bayes_preds)
        all_real.extend(real)
        all_dates.extend(dates)

    all_dates = pd.to_datetime(all_dates)
    all_bayes = np.array(all_bayes)
    all_real = np.array(all_real)

    show_real_bayes_by_points(all_dates, all_bayes, all_real)

# plot_all_bayes_prediction()
# imshow_real_bayes()
