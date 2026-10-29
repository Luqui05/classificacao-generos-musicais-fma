# Compatibilidade histórica integrada

## Contrato

`extract_features` aceita `profile='fma-2017'` e `ffmpeg_path` explícito.
O padrão `profile='current'` mantém a extração moderna anterior. A escolha
explícita evita reinterpretar silenciosamente comparações e scripts existentes.

```python
from src.audio_features import extract_features

features = extract_features(
    'faixa_completa.mp3',
    profile='fma-2017',
    ffmpeg_path=r'C:\caminho\ffmpeg.exe',
)
```

O perfil decodifica todo o MP3 para PCM16 usando `-c:a mp3` antes de `-i`,
mantém a taxa nativa e calcula a média dos canais em float32. Usa STFT com
reflect, RMS histórico, filtros chroma float64, CQT 0.5.0, coeficientes e
interpolação resampy 0.1.5, chroma com 12 bins por oitava e Hann periódica
em CENS. O esquema continua com 518 valores float32. O processamento utiliza
funções locais e não altera funções globais da Librosa.

As fontes, coeficientes e licenças estão versionados em `src/fma_2017/`.
O extrator não depende dos scripts de diagnóstico nem de arquivos em `runs/`.
O usuário fornece o executável FFmpeg separadamente. A versão validada foi
7.1, SHA256 `2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3`.
Versões Python/libs: as dos relatórios anteriores; numba 0.68.1. As dependências
de áudio diretas estão fixadas em `requirements-audio.txt`; o lock original
permanece preservado. Isso não constitui um novo lock completo de áudio.

## Regressões numéricas

As referências em `tests/fixtures/fma_complete_references.json` foram copiadas
das colunas `official` do diagnóstico original, que verificou o CSV FMA.
Não foram geradas pela saída do extrator. Incluem hashes dos MP3s completos.
O arquivo original `differences.csv` tem SHA256
`b0d5397164a7c1176507c34b71bed781095e224216c6dd6c06a21780b55ea7e9`.

```powershell
$env:FMA_AUDIO_REGRESSION='1'
.\.venv-audio\Scripts\python.exe -B -m unittest discover -s tests
```

Os testes usam os arquivos locais conhecidos. Para outra instalação, informe
`FMA_FFMPEG`, `FMA_AUDIO_30520` e `FMA_AUDIO_6406` com os caminhos respectivos.
Com o teste ativado, arquivos ausentes ou hashes diferentes causam falha.
Sem a variável, a regressão de MP3 é marcada como ignorada explicitamente.
Nenhum MP3 ou binário FFmpeg é incluído no Git ou baixado pelos testes.

Cada uma das 518 características deve satisfazer
`abs(extraído - oficial) <= 1e-6 + 1e-4 * abs(oficial)`.
As tolerâncias foram preservadas. A validação cobre duas faixas completas,
estéreo, 44100 Hz, e o decoder indicado; não comprova equivalência para toda
a FMA, todas as versões FFmpeg ou todas as taxas de amostragem.

Esta integração não treina modelos nem implementa inferência. O experimento
original continua utilizando seus dados e pipelines existentes.
