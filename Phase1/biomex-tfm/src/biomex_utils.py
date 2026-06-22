"""
biomex_utils.py - SCAFFOLD of reusable functions auto-extracted from the notebooks.

NOTE: auto-extracted (top-level defs / classes / imports / constants only).
Some module-level constants or state dependencies may be missing - review before use.
"""

# ============ IMPORTS ============
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
from scipy.signal import coherence
from scipy.signal import welch
from scipy.signal import welch as sp_welch
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import mannwhitneyu, skew as scipy_skew
from scipy.stats import pearsonr, spearmanr
from scipy.stats import spearmanr
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from speechbrain.inference.speaker import EncoderClassifier
from torch.utils.data import DataLoader, TensorDataset
import glob
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import pickle
import pywt
import seaborn as sns
import soundfile as sf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

# ============ CONSTANTES ============
PROJECT_ROOT = Path('C:\\Users\\Laura\\OneDrive\\Escritorio\\TFM_part1')
BIOMEX_ROOT = PROJECT_ROOT / 'BIOMEX' / 'BIOMEX'
METADATA_PATH = PROJECT_ROOT / 'BIOMEX_metadata.csv'
EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F8', 'F4', 'AF4']
THRESHOLD_UV = 0.00015
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
ELECTRODE_PAIRS = [('AF3', 'AF4'), ('F7', 'F8'), ('F3', 'F4'), ('FC5', 'FC6'), ('T7', 'T8'), ('P7', 'P8'), ('O1', 'O2')]
SFREQ = 128
BASELINE_DUR = 10
BASELINE_SAMP = int(BASELINE_DUR * SFREQ)
OCC_IDX = [EEG_CHANNELS.index(ch) for ch in ['O1', 'O2']]
CONDITIONS = ['REC', 'REO', 'TASK']
FRONTAL_CHANNELS = ['AF3', 'AF4', 'F7', 'F8', 'F3', 'F4']
X_113 = np.hstack([X_rec_full, X_ta_rec, iaf_aligned])
N_EPOCHS = 50
SAVE_DIR = PROJECT_ROOT / 'embeddings'
LEFT_IDX = list(range(0, 7))
RIGHT_IDX = list(range(7, 14))
METRICS = ['intra_left', 'intra_right', 'interhemispheric']
METRIC_LABELS = {'intra_left': 'Intra-left hemisphere', 'intra_right': 'Intra-right hemisphere', 'interhemispheric': 'Interhemispheric'}
SIGNIFICANT = [('intra_right_gamma', 'Intra-right γ coherence', -0.392, 0.005), ('interhemispheric_beta', 'Interhemispheric β coherence', +0.3, 0.036), ('interhemispheric_gamma', 'Interhemispheric γ coherence', -0.285, 0.047)]
BANDS_LOCAL = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
SAVE_PATH = PROJECT_ROOT / 'coherence_features_REC.csv'
FRONTAL = ['AF3', 'F7', 'F3', 'F4', 'F8', 'AF4']
FEATURE_COLS = ['iaf_O1', 'iaf_O2', 'theta_alpha_frontal', 'intra_left_gamma', 'intra_right_gamma', 'interhemispheric_gamma']
FRONTAL_PAIRS = [('AF3', 'AF4'), ('F7', 'F8'), ('F3', 'F4'), ('FC5', 'FC6')]
DIGIT_FIX = {'UN0': 'UNO', 'UNO': 'UNO', 'DOS': 'DOS', 'TRES': 'TRES', 'CUATRO': 'CUATRO', 'CINCO': 'CINCO', 'SEIS': 'SEIS', 'SIETE': 'SIETE', 'OCHO': 'OCHO', 'NUEVE': 'NUEVE', 'CERO': 'CERO', 'DIEZ': 'DIEZ'}
HTK_UNIT = 10000000.0
EPOCH_DUR = 2.0
EPOCH_SAMP = int(EPOCH_DUR * SFREQ)
STEP = EPOCH_SAMP // 2
BANDS_FP = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

# ============ FUNCIONES ============
def preprocess_eeg(edf_path, threshold_uv=150):
    """
    Full preprocessing pipeline for one BIOMEX EDF file.
    
    Steps:
        1. Load EDF and keep only the 14 EEG channels
        2. Average reference (removes DC offset)
        3. Bandpass filter 1-50 Hz (removes drifts + high freq noise)
        4. Notch filter at 50 Hz (removes Spanish powerline noise)
        5. Amplitude clipping at ±threshold_uv µV
    
    Parameters
    ----------
    edf_path    : Path or str — path to the .edf file
    threshold_uv: float — clipping threshold in microvolts (default 150)
    
    Returns
    -------
    data_clean  : np.ndarray, shape (14, n_samples), in Volts
    ch_names    : list of 14 channel names
    sfreq       : float — sampling frequency (128 Hz)
    times       : np.ndarray — time axis in seconds
    """
    EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F8', 'F4', 'AF4']
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(EEG_CHANNELS)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(l_freq=1.0, h_freq=50.0, fir_window='hamming', verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    data, times = raw[:, :]
    threshold_v = threshold_uv * 1e-06
    data = np.clip(data, -threshold_v, threshold_v)
    return (data, raw.ch_names, raw.info['sfreq'], times)

def compute_psd_features(data, sfreq=SFREQ, bands=BANDS):
    """
    Mean band power per channel using Welch's method.
    Returns array of shape (70,) and list of feature names.
    """
    features, names = ([], [])
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        freqs, psd = welch(data[ch_idx, :], fs=sfreq, nperseg=min(256, data.shape[1]))
        for band_name, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs < fhi)
            features.append(psd[mask].mean() * 1000000000000.0)
            names.append(f'{ch_name}_{band_name}')
    return (np.array(features), names)

def find_edf(subject_id, session='G10'):
    """
    Find the EDF file for a given subject and session.
    Returns the LARGEST matching EDF (most complete recording).
    """
    subject_path = BIOMEX_ROOT / subject_id / subject_id / 'EEG'
    if not subject_path.exists():
        return None
    matches = list(subject_path.glob(f'*{session}*.edf'))
    if not matches:
        return None
    return max(matches, key=lambda f: f.stat().st_size)

def compute_asymmetry_features(X_rel, feat_names, pairs=ELECTRODE_PAIRS, bands=BANDS):
    """
    Computes hemispheric asymmetry index for each electrode pair and band.
    Asymmetry = (Right - Left) / (Right + Left)
    Values in [-1, 1]: negative = left dominant, positive = right dominant.

    Parameters
    ----------
    X_rel      : np.ndarray, shape (n_subjects, n_features) — relative PSD
    feat_names : list of feature name strings e.g. ['AF3_delta', ...]

    Returns
    -------
    X_asym     : np.ndarray, shape (n_subjects, n_pairs * n_bands)
    asym_names : list of feature name strings
    """
    asym_features = []
    asym_names = []
    for left_ch, right_ch in pairs:
        for band in bands.keys():
            left_idx = feat_names.index(f'{left_ch}_{band}')
            right_idx = feat_names.index(f'{right_ch}_{band}')
            left_val = X_rel[:, left_idx]
            right_val = X_rel[:, right_idx]
            asym = (right_val - left_val) / (right_val + left_val + 1e-10)
            asym_features.append(asym)
            asym_names.append(f'asym_{left_ch}_{right_ch}_{band}')
    return (np.column_stack(asym_features), asym_names)

def train_fold(X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.LongTensor(y_val)
    model = ShallowEEGNet(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        preds = model(X_v).argmax(dim=1).numpy()
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_val, preds)

def compute_wavelet_energy(signal, wavelet='db4', level=5):
    """
    Compute Relative Wavelet Energy (RWE) for one channel.
    
    Parameters
    ----------
    signal  : np.ndarray, shape (n_samples,)
    wavelet : str — wavelet type (db4 = Daubechies 4, same as Carrión)
    level   : int — decomposition levels
    
    Returns
    -------
    rwe : np.ndarray, shape (level + 1,)
          Relative energy of [A5, D5, D4, D3, D2, D1]
          Each value is in [0, 1] and they sum to 1
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total_energy = energies.sum()
    rwe = energies / (total_energy + 1e-10)
    return rwe

def compute_wavelet_features(data, wavelet='db4', level=5):
    """
    Compute RWE features for all 14 channels.
    
    Parameters
    ----------
    data : np.ndarray, shape (14, n_samples)
    
    Returns
    -------
    features     : np.ndarray, shape (14 * (level+1),)
    feature_names: list of strings e.g. ['AF3_A5', 'AF3_D5', ...]
    """
    features = []
    feature_names = []
    coeff_names = [f'A{level}'] + [f'D{i}' for i in range(level, 0, -1)]
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        rwe = compute_wavelet_energy(data[ch_idx, :], wavelet=wavelet, level=level)
        for coeff_name, energy_val in zip(coeff_names, rwe):
            features.append(energy_val)
            feature_names.append(f'{ch_name}_{coeff_name}')
    return (np.array(features), feature_names)

def extract_condition_segments(edf_path, threshold_uv=150):
    """
    Loads a BIOMEX EDF and extracts three preprocessed EEG segments:
        REC  — eyes closed  (10 s after marker 99)
        REO  — eyes open, silent (10 s after marker 89)
        TASK — full speech/utterance section (markers 1-10)

    Parameters
    ----------
    edf_path     : Path or str
    threshold_uv : float — amplitude clipping threshold in µV (default 150)

    Returns
    -------
    dict with keys 'REC', 'REO', 'TASK'
    Each value is np.ndarray of shape (14, n_samples) in Volts,
    or None if that segment could not be extracted.

    Notes
    -----
    Preprocessing (average reference, 1-50 Hz bandpass, 50 Hz notch,
    amplitude clipping) is applied to the FULL signal before segmenting,
    so that filters do not suffer edge effects at the boundaries
    of the short 10-second segments.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    marker_raw = raw.get_data(picks=['MARKER'])[0]
    marker_data = np.round(marker_raw * 1000000.0).astype(int)
    raw.pick(EEG_CHANNELS)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(l_freq=1.0, h_freq=50.0, fir_window='hamming', verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    data, _ = raw[:, :]
    thr_v = threshold_uv * 1e-06
    data = np.clip(data, -thr_v, thr_v)
    n_samples = data.shape[1]
    transitions = np.where(np.diff(marker_data) != 0)[0] + 1
    rec_start = None
    reo_start = None
    task_start = None
    task_end = None
    for idx in transitions:
        code = marker_data[idx]
        if code == 99 and rec_start is None:
            rec_start = idx
        elif code == 89 and reo_start is None:
            reo_start = idx
        elif 1 <= code <= 10:
            if task_start is None:
                task_start = idx
            task_end = idx
    segments = {}
    if rec_start is not None:
        end = min(rec_start + BASELINE_SAMP, n_samples)
        segments['REC'] = data[:, rec_start:end]
    else:
        segments['REC'] = None
    if reo_start is not None:
        end = min(reo_start + BASELINE_SAMP, n_samples)
        segments['REO'] = data[:, reo_start:end]
    else:
        segments['REO'] = None
    if task_start is not None and task_end is not None:
        end = min(task_end + int(2 * SFREQ), n_samples)
        segments['TASK'] = data[:, task_start:end]
    else:
        segments['TASK'] = None
    return segments

def alpha_occipital(segment, sfreq=SFREQ):
    """Returns mean alpha power (8-13 Hz) at O1 and O2."""
    if segment is None:
        return None
    data_occ = segment[OCC_IDX, :]
    powers = []
    for ch in range(data_occ.shape[0]):
        freqs, psd = sp_welch(data_occ[ch, :], fs=sfreq, nperseg=min(256, data_occ.shape[1]))
        mask = (freqs >= 8) & (freqs < 13)
        powers.append(psd[mask].mean())
    return float(np.mean(powers))

def to_relative_power(df, feat_names):
    """
    Converts absolute band power to relative power (per subject).
    Each subject's feature vector is divided by its total power,
    so all values are in [0, 1] and sum to 1 per subject.
    """
    df_out = df.copy()
    X = df[feat_names].values
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    df_out[feat_names] = X / row_sum
    return df_out

def compute_theta_alpha_ratio(df, feat_names):
    """
    Computes theta/alpha ratio for each frontal channel.
    Higher ratio = more theta relative to alpha = older brain signature.
    Returns array (n_subjects, n_frontal_channels) and feature names.
    """
    ratios, names = ([], [])
    for ch in FRONTAL_CHANNELS:
        theta_idx = feat_names.index(f'{ch}_theta')
        alpha_idx = feat_names.index(f'{ch}_alpha')
        theta_val = df[feat_names].values[:, theta_idx]
        alpha_val = df[feat_names].values[:, alpha_idx]
        ratio = theta_val / (alpha_val + 1e-10)
        ratios.append(ratio)
        names.append(f'ta_ratio_{ch}')
    return (np.column_stack(ratios), names)

def compute_iaf(segment, sfreq=SFREQ, channels=['O1', 'O2'], fmin=7.0, fmax=14.0):
    """
    Computes Individual Alpha Frequency (IAF) per channel.
    IAF = frequency of peak power within the extended alpha range.

    Parameters
    ----------
    segment  : np.ndarray, shape (14, n_samples)
    channels : list of channel names to compute IAF on
    fmin/fmax: search range in Hz (slightly wider than 8-13
               to catch edge cases in older subjects)

    Returns
    -------
    iaf_values : list of float, one per channel
    iaf_names  : list of feature name strings
    """
    iaf_values = []
    iaf_names = []
    for ch_name in channels:
        ch_idx = EEG_CHANNELS.index(ch_name)
        freqs, psd = sp_welch(segment[ch_idx, :], fs=sfreq, nperseg=min(256, segment.shape[1]))
        alpha_mask = (freqs >= fmin) & (freqs <= fmax)
        peak_freq = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        iaf_values.append(peak_freq)
        iaf_names.append(f'iaf_{ch_name}')
    return (iaf_values, iaf_names)

def extract_digit_epochs(edf_path, epoch_len_s=2.0, threshold_uv=150):
    """
    Extracts one EEG epoch per digit utterance from a BIOMEX EDF.

    Parameters
    ----------
    edf_path    : Path or str
    epoch_len_s : float — epoch length in seconds (default 2.0)

    Returns
    -------
    epochs      : np.ndarray, shape (n_digits, 14, n_samples)
    labels      : list of int — digit code (1-10) for each epoch
    """
    epoch_samples = int(epoch_len_s * SFREQ)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    marker_raw = raw.get_data(picks=['MARKER'])[0]
    marker_data = np.round(marker_raw * 1000000.0).astype(int)
    raw.pick(EEG_CHANNELS)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(l_freq=1.0, h_freq=50.0, fir_window='hamming', verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    data, _ = raw[:, :]
    thr_v = threshold_uv * 1e-06
    data = np.clip(data, -thr_v, thr_v)
    n_samples = data.shape[1]
    transitions = np.where(np.diff(marker_data) != 0)[0] + 1
    epochs, labels = ([], [])
    for idx in transitions:
        code = marker_data[idx]
        if 1 <= code <= 10:
            end = idx + epoch_samples
            if end <= n_samples:
                epochs.append(data[:, idx:end])
                labels.append(code)
    if len(epochs) == 0:
        return (None, None)
    return (np.stack(epochs), labels)

def compute_coherence_matrix(segment, sfreq=SFREQ, bands=BANDS, nperseg=128):
    """
    Band-averaged coherence between every pair of EEG channels.

    Parameters
    ----------
    segment : np.ndarray, shape (14, n_samples) — one clean EEG segment
    sfreq   : float — sampling frequency (128 Hz)
    bands   : dict — {band_name: (f_low, f_high)}
    nperseg : int  — window length in samples for Welch averaging

    Returns
    -------
    coh : dict {band_name: np.ndarray (14, 14)}
          Symmetric coherence matrices, diagonal forced to 1.
    """
    n_ch = segment.shape[0]
    nperseg_eff = min(nperseg, segment.shape[1])
    coh = {band: np.eye(n_ch) for band in bands}
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, Cxy = coherence(segment[i], segment[j], fs=sfreq, nperseg=nperseg_eff)
            for band, (flo, fhi) in bands.items():
                mask = (f >= flo) & (f < fhi)
                val = Cxy[mask].mean()
                coh[band][i, j] = val
                coh[band][j, i] = val
    return coh

def age_group(age):
    if age < 30:
        return 'Young (<30)'
    if age <= 45:
        return 'Middle (30–45)'
    return 'Older (>45)'

def compute_psd_features_rec(segment, sfreq=SFREQ, bands=BANDS_LOCAL):
    """Mean band power per channel (Welch) for one segment."""
    features, names = ([], [])
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        f, psd = welch(segment[ch_idx, :], fs=sfreq, nperseg=min(256, segment.shape[1]))
        for band, (flo, fhi) in bands.items():
            mask = (f >= flo) & (f < fhi)
            features.append(psd[mask].mean() * 1000000000000.0)
            names.append(f'{ch_name}_{band}')
    return (np.array(features), names)

def extract_age_features(segment, sfreq=SFREQ):
    """IAF (O1,O2) + frontal theta/alpha ratio for one REC segment."""
    feats = {}
    for ch in ['O1', 'O2']:
        idx = EEG_CHANNELS.index(ch)
        f, psd = welch(segment[idx], fs=sfreq, nperseg=min(256, segment.shape[1]))
        amask = (f >= 7) & (f <= 14)
        feats[f'iaf_{ch}'] = f[amask][np.argmax(psd[amask])]
    ratios = []
    for ch in FRONTAL:
        idx = EEG_CHANNELS.index(ch)
        f, psd = welch(segment[idx], fs=sfreq, nperseg=min(256, segment.shape[1]))
        theta = psd[(f >= 4) & (f < 8)].mean()
        alpha = psd[(f >= 8) & (f < 13)].mean()
        ratios.append(theta / (alpha + 1e-12))
    feats['theta_alpha_frontal'] = np.mean(ratios)
    return feats

def alpha_power(segment, ch_name, sfreq=SFREQ):
    """Mean alpha (8-13 Hz) power for one channel."""
    idx = EEG_CHANNELS.index(ch_name)
    f, psd = welch(segment[idx], fs=sfreq, nperseg=min(256, segment.shape[1]))
    return psd[(f >= 8) & (f < 13)].mean()

def parse_lab(lab_path):
    """
    Parse an HTK .lab file into a list of (start_s, end_s, label).
    Times converted to seconds. Labels normalised. Silence kept
    (flagged) so we can verify coverage, but caller can filter it.
    """
    segments = []
    with open(lab_path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            start = float(parts[0]) / HTK_UNIT
            end = float(parts[1]) / HTK_UNIT
            label = parts[2].strip().upper()
            label = DIGIT_FIX.get(label, label)
            segments.append((start, end, label))
    return segments

def extract_speech(wav_path, lab_path, sr_expected=16000, pad_s=0.05):
    """
    Load a wav and return only the concatenated spoken-digit audio
    (silence removed), plus the list of digits spoken in order.

    pad_s: small padding (s) added around each segment so we don't
           clip the onset/offset of each digit.
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    segments = parse_lab(lab_path)
    speech_chunks = []
    digits = []
    for start, end, label in segments:
        if label == 'SILENCIO':
            continue
        i0 = max(0, int((start - pad_s) * sr))
        i1 = min(len(audio), int((end + pad_s) * sr))
        speech_chunks.append(audio[i0:i1])
        digits.append(label)
    speech = np.concatenate(speech_chunks) if speech_chunks else np.array([])
    return (speech, digits, sr)

def embed_speech(speech, sr=16000):
    """Return a 192-dim ECAPA embedding for a 1-D speech array."""
    wav = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = spk_encoder.encode_batch(wav)
    emb = emb.squeeze().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-12)

def cosine(a, b):
    return float(np.dot(a, b))

def clip_paths(sid, n):
    vp = BIOMEX_ROOT / sid / sid / 'Voice'
    return (vp / f'{sid}_01G04_{n}.wav', vp / f'{sid}_01G04_{n}.lab')

def subject_voice_embedding(sid, session='G10'):
    """
    Average ECAPA embedding over all valid clips of one session.
    Returns (embedding_192, n_clips_used) or (None, 0) if no audio.
    """
    vp = BIOMEX_ROOT / sid / sid / 'Voice'
    if not vp.exists():
        return (None, 0)
    clip_embs = []
    for wav in sorted(vp.glob(f'*{session}*.wav')):
        lab = wav.with_suffix('.lab')
        if not lab.exists():
            continue
        try:
            speech, digits, sr = extract_speech(wav, lab)
            if len(speech) < 0.3 * sr:
                continue
            clip_embs.append(embed_speech(speech, sr))
        except Exception:
            continue
    if not clip_embs:
        return (None, 0)
    mean_emb = np.mean(clip_embs, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
    return (mean_emb, len(clip_embs))

def alpha_power_arr(data, ch_name, sfreq=SFREQ):
    """Mean alpha (8-13 Hz) power for one channel from a data array (14, n)."""
    idx = EEG_CHANNELS.index(ch_name)
    f, psd = welch(data[idx], fs=sfreq, nperseg=min(256, data.shape[1]))
    return psd[(f >= 8) & (f < 13)].mean()

def epoch_faa_dist(segment, pairs=FRONTAL_PAIRS, sfreq=SFREQ):
    """
    Split segment into 2-second epochs (50% overlap) and compute FAA per epoch.
    Returns dict: 'L_R' -> np.array of per-epoch FAA values.
    """
    n = segment.shape[1]
    faa_by_pair = {f'{l}_{r}': [] for l, r in pairs}
    start = 0
    while start + EPOCH_SAMP <= n:
        epoch = segment[:, start:start + EPOCH_SAMP]
        for l, r in pairs:
            pL = alpha_power_arr(epoch, l, sfreq)
            pR = alpha_power_arr(epoch, r, sfreq)
            faa_by_pair[f'{l}_{r}'].append(np.log(pR + 1e-12) - np.log(pL + 1e-12))
        start += STEP
    return {k: np.array(v) for k, v in faa_by_pair.items()}

def psd_features(segment, sfreq=SFREQ):
    """70-dim PSD feature vector (5 bands x 14 channels, log-power)."""
    feats = []
    for ch in range(segment.shape[0]):
        freqs, psd = welch(segment[ch], fs=sfreq, nperseg=min(256, segment.shape[1]))
        for lo, hi in BANDS_FP.values():
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(np.log1p(psd[mask].mean()))
    return np.array(feats)

# ============ CLASES ============
class ShallowEEGNet(nn.Module):

    def __init__(self, input_dim, hidden1=64, hidden2=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(0.4), nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden2, n_classes))

    def forward(self, x):
        return self.net(x)

class EEGNet(nn.Module):

    def __init__(self, n_classes=51, n_channels=14, n_samples=256, F1=8, D=2, F2=16, embed_dim=32, dropout=0.5):
        """
        Parameters
        ----------
        n_classes  : int — number of subjects to identify
        n_channels : int — EEG channels (14)
        n_samples  : int — time samples per epoch (256)
        F1         : int — number of temporal filters
        D          : int — depth multiplier for spatial filters
        F2         : int — number of separable filters (= F1 * D)
        embed_dim  : int — embedding vector size
        dropout    : float — dropout rate
        """
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(F1))
        self.block2 = nn.Sequential(nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout))
        self.block3 = nn.Sequential(nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False), nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d(kernel_size=(1, 8)), nn.Dropout(dropout))
        self._flat_size = self._get_flat_size(n_channels, n_samples, F1, D, F2)
        self.embedding = nn.Linear(self._flat_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def _get_flat_size(self, n_channels, n_samples, F1, D, F2):
        """Pass a dummy tensor to compute the flattened size."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.view(1, -1).shape[1]

    def forward(self, x, return_embedding=False):
        """
        Parameters
        ----------
        x                : tensor, shape (batch, 14, 256)
        return_embedding : if True, return 32-dim embedding instead
                           of class logits — used at inference time
                           to extract the neural fingerprint
        """
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        embed = F.elu(self.embedding(x))
        if return_embedding:
            return embed
        return self.classifier(embed)
