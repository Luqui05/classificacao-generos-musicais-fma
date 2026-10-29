# Recursos históricos FMA

Fontes `constantq.py` e `filters.py`: Librosa 0.5.0, licença ISC anexa.
Coeficientes `kaiser_fast.npz` e `kaiser_best.npz`: resampy 0.1.5,
licença ISC anexa. Arquivos preservados sem modificações.

Hashes dos pacotes e URLs: `validation/audio_features/diagnostic_sources.json`.
Os hashes dos arquivos são verificados por `src/_fma_legacy.py` antes do uso.
As fontes são lidas por AST; somente as funções numéricas selecionadas são
compiladas. Não há importação ou instalação das versões antigas das bibliotecas.

`src/_fma_resample.py` é uma transcrição mono do laço de `resampy/interp.pyx`
0.1.5, com verificação adicional de índices. Atribuição e licença resampy
aplicam-se também a essa adaptação.
