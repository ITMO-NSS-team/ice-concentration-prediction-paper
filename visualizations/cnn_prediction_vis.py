from datetime import timedelta
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from Converters import EncoderBased


def get_real_images(dates_range):
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


def return_ssim_for_massive(massive, real_massive):
    metrics_list = []
    for step in range(massive.shape[0]):
        pred = massive[step]
        real = real_massive[step]
        ssim_metric = 1 - structural_similarity(pred, real, data_range=1)
        metrics_list.append(ssim_metric)
    return np.array(metrics_list)


def return_mae_for_massive(massive, real_massive):
    return np.mean(abs(massive - real_massive), axis=(1, 2))


def get_cnn_prediction(dates_range, model_path):
    model = EncoderBased(model_path)
    start_date = dates_range.tolist()[0]
    dates_of_pre_history = pd.date_range(start_date - timedelta(days=7 * 156), periods=156, freq='7D')
    dates_of_pre_history = [d.strftime('%Y%m%d') for d in dates_of_pre_history]
    input_data = []
    for date in dates_of_pre_history:
        try:
            matrix = np.load(f'../matrices/osisaf/train/osi_iceconc_{date}.npy')
        except Exception:
            matrix = np.load(f'../matrices/osisaf/test/osi_iceconc_{date}.npy')
        input_data.append(matrix)
    input_data = np.array(input_data)
    input_data = torch.Tensor(input_data)
    input_data = input_data.to("cuda")
    prediction = model.encoder(input_data)
    prediction = prediction.cpu().detach().numpy()
    dates_range = dates_range[::7]
    final_index = 52 if len(dates_range) > 52 else len(dates_range)
    prediction = prediction[0:final_index, :, :]
    dates_range = dates_range[:final_index]
    return prediction, dates_range


def return_binary_massive(massive, threshold):
    massive[massive > threshold] = 1
    massive[massive <= threshold] = 0
    return massive


def return_ice_area_for_massive(massive, threshold):
    massive[massive > threshold] = 1
    massive[massive <= threshold] = 0
    areas = []
    for i in range(massive.shape[0]):
        image = massive[i]
        unique, counts = np.unique(image, return_counts=True)
        area = image.shape[0]*image.shape[1] - dict(zip(unique, counts))[0]
        area_in_km = area*14*14
        areas.append(area_in_km)
    return areas


def mae(prediction, real):
    #return round(float(np.mean(abs(np.array(prediction) - np.array(real)))), 3)
    return round(float(np.mean(abs(np.array(prediction) - np.array(real)))))


def ssim(prediction, real):
    return round(structural_similarity(prediction, real, data_range=1), 3)


def accuracy(prediction, real):
    f = 125*125
    diff = prediction-real
    unique, counts = np.unique(diff, return_counts=True)
    right_calculated = dict(zip(unique, counts))[0]
    return round(right_calculated/f, 3)

def imshow_real_l1_ssmi():
    start_time = '20150101'
    end_time = '20151231'
    prediction_dates = pd.Series(pd.date_range(start=start_time, end=end_time, freq='D'))

    l1_model = f'../fitted_models/long_term_multioutput/osi/osi_1990-2012_lag156_for52w.pkl'
    ssim_model = f'../fitted_models/long_term_multioutput/osi/osi_1990-2012_lag156_for52w_ssim.pkl'

    l1_preds, _ = get_cnn_prediction(prediction_dates, l1_model)
    ssim_preds, _ = get_cnn_prediction(prediction_dates, ssim_model)
    real, dates = get_real_images(prediction_dates)

    for i in range(real.shape[0]):
        plt.rcParams["figure.figsize"] = (10, 8)
        f, axarr = plt.subplots(2, 3)
        axarr[0, 0].imshow(real[i], vmin=0, vmax=1)
        axarr[0, 0].title.set_text('Real data')
        axarr[0, 1].imshow(l1_preds[i], vmin=0, vmax=1)
        axarr[0, 1].title.set_text(f'CNN L1Loss\nMAE={mae(l1_preds[i], real[i])}\n'
                                   f'SSIM={ssim(l1_preds[i], real[i])}')
        img = axarr[0, 2].imshow(ssim_preds[i], vmin=0, vmax=1)
        axarr[0, 2].title.set_text(f'CNN SSIM loss\nMAE={mae(ssim_preds[i], real[i])}\n'
                                   f'SSIM={ssim(ssim_preds[i], real[i])}')

        real_binary = return_binary_massive(real[i], 0.8)
        l1_binary = return_binary_massive(l1_preds[i], 0.8)
        ssim_binary = return_binary_massive(ssim_preds[i], 0.65)
        axarr[1, 0].imshow(real_binary, vmin=0, vmax=1)
        axarr[1, 0].title.set_text('Real data ice edge')
        axarr[1, 1].imshow(l1_binary, vmin=0, vmax=1)
        axarr[1, 1].title.set_text(f'CNN L1Loss ice edge\naccuracy={accuracy(l1_binary, real_binary)}')
        axarr[1, 2].imshow(ssim_binary, vmin=0, vmax=1)
        axarr[1, 2].title.set_text(f'CNN SSIM loss ice edge\naccuracy={accuracy(ssim_binary, real_binary)}')
        c = plt.colorbar(img, ax=axarr[0, 2])
        c.set_label('Ice concentration', rotation=90)
        for k in range(2):
            for j in range(3):
                axarr[k, j].axis('off')

        f.suptitle(f'Prediction date - {dates[i]}')

        plt.savefig(f'cnn_preds/week_{i}.png', dpi=300)
        plt.show()


def get_ice_area_prediction(dates_range):
    l1_model = f'../fitted_models/long_term_multioutput/osi/osi_1990-2012_lag156_for52w.pkl'
    ssim_model = f'../fitted_models/long_term_multioutput/osi/osi_1990-2012_lag156_for52w_ssim.pkl'

    l1_preds, _ = get_cnn_prediction(dates_range, l1_model)
    ssim_preds, _ = get_cnn_prediction(dates_range, ssim_model)
    real, dates = get_real_images(dates_range)

    l1_area = return_ice_area_for_massive(l1_preds, 0.8)
    ssim_area = return_ice_area_for_massive(ssim_preds, 0.7)
    real_area = return_ice_area_for_massive(real, 0.8)

    return dates, l1_area, ssim_area, real_area


def plot_ice_area(dates, l1_area, ssim_area, real_area):
    dates = pd.to_datetime(dates)
    plt.plot(dates, l1_area, label=f'CNN L1Loss, MAE={mae(l1_area, real_area)} km^2')
    plt.plot(dates, ssim_area, label=f'CNN SSIM Loss, MAE={mae(ssim_area, real_area)} km^2')
    plt.plot(dates, real_area, label='Real data')
    plt.ylabel('Ice area (km^2)')
    plt.title(f'(b) Ice area (km) dynamics')
    plt.legend()
    plt.savefig('cnn_ice_area.png', dpi=300)
    plt.show()


def plot_ice_area_for_all_period():
    years = range(2013, 2018)
    all_l1 = []
    all_ssim = []
    all_real = []
    all_dates = []
    for year in years:
        prediction_dates = pd.Series(pd.date_range(start=f'{year}0101', end=f'{year}1231', freq='D'))
        dates, l1_area, ssim_area, real_area = get_ice_area_prediction(prediction_dates)
        all_l1.extend(l1_area)
        all_ssim.extend(ssim_area)
        all_real.extend(real_area)
        all_dates.extend(dates)

    plot_ice_area(all_dates, all_l1, all_ssim, all_real)


def get_accuracy_prediction(dates_range):
    l1_model = f'../fitted_models/long_term_multioutput/osi/osi_1990-2012_lag156_for52w.pkl'
    ssim_model = f'../fitted_models/long_term_multioutput/osi/osi_1990-2012_lag156_for52w_ssim.pkl'

    l1_preds, _ = get_cnn_prediction(dates_range, l1_model)
    ssim_preds, _ = get_cnn_prediction(dates_range, ssim_model)
    real, dates = get_real_images(dates_range)

    real = return_binary_massive(real, 0.8)
    l1_preds = return_binary_massive(l1_preds, 0.8)
    ssim_preds = return_binary_massive(ssim_preds, 0.7)

    l1_all = []
    ssim_all = []
    for i in range(real.shape[0]):
        l1_accuracy = accuracy(l1_preds[i], real[i])
        ssim_accuracy = accuracy(ssim_preds[i], real[i])
        l1_all.append(l1_accuracy)
        ssim_all.append(ssim_accuracy)
    return dates, l1_all, ssim_all


def plot_all_accuracy():
    years = range(2013, 2018)
    all_l1 = []
    all_ssim = []
    all_dates = []
    for year in years:
        prediction_dates = pd.Series(pd.date_range(start=f'{year}0101', end=f'{year}1231', freq='D'))
        dates, l1_val, ssim_val = get_accuracy_prediction(prediction_dates)
        all_l1.extend(l1_val)
        all_ssim.extend(ssim_val)
        all_dates.extend(dates)

    dates = pd.to_datetime(all_dates)
    plt.ylabel('Accuracy')
    plt.plot(dates, all_l1, label=f'CNN L1Loss')
    plt.plot(dates, all_ssim, label=f'CNN SSIM Loss')
    plt.title(f'(a) Accuracy dynamics')
    plt.legend()
    plt.savefig('cnn_accuracy.png', dpi=300)
    plt.show()

#plot_ice_area_for_all_period()
#imshow_real_l1_ssmi()
#plot_all_accuracy()
