"""Extração de características de áudio com o esquema FMA para um único arquivo.

Os cálculos deste módulo seguem o features.py do repositório oficial
do Free Music Archive. A saída é uma Series float32 cujos nomes de coluna
planos e cuja ordem correspondem a fma_small_features.csv. A equivalência
numérica com os valores publicados ainda requer validação com áudios oficiais.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


FEATURE_SIZES: Final[dict[str, int]] = {
    "chroma_stft": 12,
    "chroma_cqt": 12,
    "chroma_cens": 12,
    "tonnetz": 6,
    "mfcc": 20,
    "rmse": 1,
    "zcr": 1,
    "spectral_centroid": 1,
    "spectral_bandwidth": 1,
    "spectral_contrast": 7,
    "spectral_rolloff": 1,
}
STATISTICS: Final[tuple[str, ...]] = (
    "mean",
    "std",
    "skew",
    "kurtosis",
    "median",
    "min",
    "max",
)
_FEATURE_SORT_ORDER: Final[tuple[str, ...]] = tuple(sorted(FEATURE_SIZES))
_STATISTIC_SORT_ORDER: Final[tuple[str, ...]] = tuple(sorted(STATISTICS))

# pandas.MultiIndex.sort_values() no script oficial ordena lexicograficamente
# os componentes das tuplas de strings, incluindo o número de característica
# com dois dígitos.
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{feature}_{statistic}_{number:02d}"
    for feature in _FEATURE_SORT_ORDER
    for statistic in _STATISTIC_SORT_ORDER
    for number in range(1, FEATURE_SIZES[feature] + 1)
)
EXPECTED_FEATURE_COUNT: Final[int] = 518

if len(FEATURE_NAMES) != EXPECTED_FEATURE_COUNT:  # pragma: no cover - invariante
    raise RuntimeError(f"Erro interno: esperadas 518 features, obtidas {len(FEATURE_NAMES)}")


def validate_features(features: pd.Series) -> pd.Series:
    """Valida e normaliza um vetor extraído no padrão FMA.

    Os rótulos devem estar exatamente na mesma ordem da saída oficial do FMA;
    vetores contendo NaN ou infinito são rejeitados em vez de serem repassados
    para um classificador.
    """
    if not isinstance(features, pd.Series):
        raise TypeError("O vetor de características deve ser uma pandas.Series.")
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Esperadas {EXPECTED_FEATURE_COUNT} características; recebidas {len(features)}."
        )
    if tuple(features.index) != FEATURE_NAMES:
        raise ValueError("A ordem ou os nomes das características não correspondem ao padrão FMA.")

    try:
        values = features.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise TypeError("Todas as características devem ser numéricas.") from error
    if not np.isfinite(values).all():
        invalid = [FEATURE_NAMES[i] for i in np.flatnonzero(~np.isfinite(values))[:5]]
        raise ValueError(f"Características não finitas encontradas: {', '.join(invalid)}")

    with np.errstate(over="ignore", invalid="ignore"):
        normalized = features.astype(np.float32)
    if not np.isfinite(normalized.to_numpy()).all():
        raise ValueError("Características fora do intervalo finito de float32.")
    return normalized


def _feature_statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula as sete estatísticas utilizadas pelo script original do FMA."""
    from scipy import stats

    return {
        "mean": np.mean(values, axis=1),
        "std": np.std(values, axis=1),
        "skew": stats.skew(values, axis=1),
        "kurtosis": stats.kurtosis(values, axis=1),
        "median": np.median(values, axis=1),
        "min": np.min(values, axis=1),
        "max": np.max(values, axis=1),
    }


def extract_features(audio_path: str | Path) -> pd.Series:
    """Extrai as 518 estatísticas do FMA a partir de um arquivo de áudio (incluindo MP3).

    O carregamento preserva a taxa de amostragem nativa do arquivo e converte para mono,
    exatamente como o código oficial do FMA faz. A biblioteca librosa é importada sob
    demanda (lazy import) para que o restante do experimento do TCC possa continuar
    executando sem dependências de áudio.

    Raises:
        FileNotFoundError: se audio_path não apontar para um arquivo.
        RuntimeError: se a librosa não estiver instalada.
        ValueError: se o arquivo estiver vazio/inválido, os cálculos falharem ou
            qualquer estatística resultante for não finita.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {path}")

    try:
        import librosa
    except ImportError as error:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "Librosa não está instalada. Consulte audio_features.md."
        ) from error

    computed: dict[str, dict[str, np.ndarray]] = {}

    def record(name: str, values: np.ndarray) -> None:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError(f"A saída de {name} tem formato inválido: {array.shape}.")
        expected_bins = FEATURE_SIZES[name]
        if array.shape[0] != expected_bins:
            raise ValueError(
                f"A saída de {name} tem {array.shape[0]} bandas; esperadas {expected_bins}."
            )
        computed[name] = _feature_statistics(array)

    try:
        y, sample_rate = librosa.load(path, sr=None, mono=True)
        if y.size == 0:
            raise ValueError("O arquivo de áudio não contém amostras.")

        record(
            "zcr",
            librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512),
        )

        cqt = np.abs(
            librosa.cqt(
                y,
                sr=sample_rate,
                hop_length=512,
                bins_per_octave=12,
                n_bins=7 * 12,
                tuning=None,
            )
        )
        if cqt.shape[0] != 7 * 12:
            raise ValueError(f"CQT inesperado: {cqt.shape[0]} bandas, esperadas 84.")
        expected_frames = int(np.ceil(len(y) / 512))
        if not expected_frames <= cqt.shape[1] <= expected_frames + 1:
            raise ValueError(f"Quantidade inesperada de quadros CQT: {cqt.shape[1]}.")

        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        record("chroma_cqt", chroma_cqt)
        chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        record("chroma_cens", chroma_cens)
        record("tonnetz", librosa.feature.tonnetz(chroma=chroma_cens))

        del cqt
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        if stft.shape[0] != 1 + 2048 // 2:
            raise ValueError(f"STFT inesperada: {stft.shape[0]} bandas, esperadas 1025.")
        if not expected_frames <= stft.shape[1] <= expected_frames + 1:
            raise ValueError(f"Quantidade inesperada de quadros STFT: {stft.shape[1]}.")
        del y

        record("chroma_stft", librosa.feature.chroma_stft(S=stft**2, n_chroma=12))

        # Adapta o nome da API mantendo a magnitude da STFT como entrada.
        # A equivalência numérica entre versões de rmse/rms precisa ser validada.
        rms = getattr(librosa.feature, "rmse", None) or librosa.feature.rms
        record("rmse", rms(S=stft))
        record("spectral_centroid", librosa.feature.spectral_centroid(S=stft))
        record("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft))
        record("spectral_contrast", librosa.feature.spectral_contrast(S=stft, n_bands=6))
        record("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft))

        mel = librosa.feature.melspectrogram(sr=sample_rate, S=stft**2)
        del stft
        record("mfcc", librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))
    except Exception as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("O arquivo", "CQT", "STFT", "A saída", "Quantidade")
        ):
            raise
        raise ValueError(f"Falha ao extrair características de {path.name}: {error}") from error

    values = {
        f"{feature}_{statistic}_{index:02d}": computed[feature][statistic][index - 1]
        for feature in _FEATURE_SORT_ORDER
        for statistic in _STATISTIC_SORT_ORDER
        for index in range(1, FEATURE_SIZES[feature] + 1)
    }
    result = pd.Series(values, dtype=np.float32, name=path.stem)
    return validate_features(result)
