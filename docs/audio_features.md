# Extração de características de áudio

O módulo `src/audio_features.py` recebe um caminho de áudio (inclusive MP3) e
retorna uma `pandas.Series` com 518 valores `float32`. Os nomes e a ordem das
colunas correspondem aos atributos de `data/fma_small_features.csv`, sem
`track_id` nem `genre`. O nome da Series é o nome do arquivo sem a extensão.

**Estado da validação:** o perfil explícito `fma-2017` reproduziu os 518 valores
oficiais nas faixas completas 030520 e 006406 dentro de
`abs(extraído - oficial) <= 1e-6 + 1e-4 * abs(oficial)`.
A validação é limitada a esses arquivos e ao ambiente documentado; não garante
equivalência em outros áudios ou trechos. Os diagnósticos iniciais dos trechos
Small permanecem preservados no [histórico](../validation/audio_features/STATUS.md).
O perfil moderno `current` é o padrão e não possui essa garantia numérica.

Os cinco pipelines finais foram salvos e a [inferência](inference.md) está
implementada. A [avaliação complementar](unseen_fma.md) processou 40 faixas fora
do treinamento, da mesma FMA. Nesses trechos de 30 segundos, somente 21 a 51
dos 518 atributos por faixa ficaram na tolerância em relação aos oficiais.
Consulte a [comparação pareada e seus limites](../validation/unseen_fma/OFFICIAL_COMPARISON.md).

## Instalação opcional e isolada

`requirements.txt` e `requirements-lock.txt` mantêm o conteúdo original do
experimento, sem dependências de áudio. `requirements-audio.txt` fixa a
dependências diretas `librosa==0.11.0`, `numba==0.68.1` e `soundfile==0.14.0`
e usa o lock original como restrição para
as bibliotecas compartilhadas. Isso impede atualizações dessas versões;
uma combinação incompatível deve falhar na resolução, em vez de atualizá-las.

Execute a partir da raiz do projeto, em um ambiente separado do experimento:

```powershell
py -3.11 -m venv .venv-audio
.\.venv-audio\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-audio\Scripts\python.exe -m pip install -r requirements-audio.txt
.\.venv-audio\Scripts\python.exe -m pip check
```

Em Linux/macOS, use `python3.11 -m venv .venv-audio` e
`.venv-audio/bin/python` nos comandos seguintes.

O arquivo opcional não é um lock completo das dependências transitivas de
áudio. Para reproduzir uma comparação numérica, registre também
Python, todas as versões instaladas, sistema operacional, decodificador e
parâmetros utilizados, em um registro separado do lock original.

No perfil `current`, o suporte a MP3 depende do backend disponível. Se houver falha
de decodificação, verifique o erro original encadeado; conforme o backend,
pode ser necessário disponibilizar FFmpeg no `PATH`. Os testes unitários
com mocks não verificam a instalação de um decodificador. O perfil `fma-2017`
exige o caminho explícito do FFmpeg; sua versão validada e seu hash estão no
[guia de integração](../validation/audio_features/INTEGRATION.md).

## Uso do extrator

```python
from src.audio_features import extract_features

features = extract_features("caminho/para/faixa.mp3")
print(features.shape)  # (518,)
```

O carregamento usa `sr=None` e `mono=True`: preserva a taxa de amostragem e
converte para mono. O arquivo inteiro é analisado, sem corte, normalização de
volume ou seleção automática de trecho. No perfil histórico, a decodificação
usa FFmpeg `mp3` de ponto fixo para PCM16 e a média dos canais, preservando a
taxa nativa. Uma gravação completa e um trecho de 30 segundos não são entradas
numericamente equivalentes, mesmo quando compartilham o `track_id`.

O vetor é rejeitado se tiver nomes/ordem inesperados, valores NaN/infinito ou
valores que transbordem ao converter para `float32`. Uma faixa silenciosa,
constante ou curta demais pode gerar estatísticas indefinidas ou falhas nas
transformadas. Nesses casos, a extração falha, sem preencher atributos com
zeros ou retornar um vetor parcial.

## Limites da compatibilidade com a FMA

O código segue a sequência do
[`features.py` oficial](https://github.com/mdeff/fma/blob/master/features.py),
incluindo Tonnetz a partir de chroma CENS e as sete estatísticas agregadas.
As [dependências históricas da FMA](https://github.com/mdeff/fma/blob/master/requirements.txt)
indicam Librosa 0.5.0, com ressalvas dos autores sobre esse registro.

Usar Librosa 0.11.0 não garante os mesmos resultados: houve mudanças no
preenchimento das bordas, normalização de RMS e outros cálculos/padrões.
O fallback de `rmse` para `rms` adapta o nome da API, sem comprovar igualdade
numérica. Consulte o [histórico do Librosa](https://librosa.org/doc/0.11.0/changelog.html).
Algumas chamadas espectrais omitem `sr`, como no script oficial; alterar
isso exige comparação com os valores publicados, mesmo que pareça uma correção.

O comparador e a lista de MP3s necessários estão em
[audio_validation.md](audio_validation.md). Ele compara os 518 valores de MP3s
oficiais, identificados por `track_id`, com as respectivas linhas do CSV e
resume diferenças por família e estatística. A primeira execução com áudio
real está analisada no acompanhamento acima. Correspondência de colunas e
testes com mocks não substituem essa comparação.

## Testes automatizados

A suíte completa exige o ambiente de áudio, incluindo Librosa. Ela reúne
contratos com mocks, controles analíticos com sinais sintéticos, testes de
segurança dos pipelines e regressões reais opcionais. O teste de esquema lê
o cabeçalho de `data/fma_small_features.csv`; execute na raiz do projeto.

```powershell
.\.venv-audio\Scripts\python.exe -B -m unittest discover -s tests -v
```

A opção `-B` evita gravar bytecode. Sem `FMA_AUDIO_REGRESSION=1`, a regressão
real é explicitamente ignorada; isso não comprova extração numérica de MP3.
Com os dois arquivos completos e FFmpeg disponíveis, execute:

```powershell
$env:FMA_AUDIO_REGRESSION='1'
.\.venv-audio\Scripts\python.exe -B -m unittest discover -s tests -v
```

Os caminhos e as variáveis `FMA_FFMPEG`, `FMA_AUDIO_30520` e `FMA_AUDIO_6406`
estão no [guia de integração](../validation/audio_features/INTEGRATION.md).
Na revisão de 02/11/2026, essa execução passou com 61 testes, sem falhas,
erros ou testes ignorados. Esse total é um registro daquela revisão.

Cobertura de comportamento:

- Nomes e ordem comparados ao cabeçalho real do CSV de treinamento.
- Sete estatísticas comparadas a valores calculados analiticamente.
- Validação de tipo, tamanho, rótulos, finitude e transbordamento em `float32`.
- Importação sem Librosa e mensagem para dependência ausente.
- Montagem dos 518 atributos, taxa nativa, mono, magnitude/potência e uso de CENS
  no Tonnetz, com respostas controladas de um módulo de áudio simulado.
- Arquivo ausente, diretório, áudio vazio, falha de decodificação, formas
  inválidas e estatísticas não finitas, sem vetor parcial.

Os mocks verificam a integração entre chamadas e a montagem da saída. Não
testam os algoritmos internos do Librosa, a decodificação MP3 ou a equivalência
numérica com a FMA. Os testes não escrevem em `data/` nem em `results/`.
# Perfil histórico integrado

Para reproduzir o perfil validado nas faixas completas 030520 e 006406,
use `extract_features(path, profile='fma-2017', ffmpeg_path=...)`.
O comportamento anterior continua disponível e é o padrão `current`.
Consulte [contrato e regressões](../validation/audio_features/INTEGRATION.md).
