# Extração de características de áudio

O módulo `src/audio_features.py` recebe um caminho de áudio (inclusive MP3) e
retorna uma `pandas.Series` com 518 valores `float32`. Os nomes e a ordem das
colunas correspondem aos atributos de `data/fma_small_features.csv`, sem
`track_id` nem `genre`. O nome da Series é o nome do arquivo sem a extensão.

**Estado da validação:** o esquema e o contrato do extrator têm testes
automatizados. A equivalência numérica com os valores publicados pela FMA
ainda não foi validada com áudios oficiais. Não há treinamento nem pipeline
de inferência implementados nesta extensão.

## Instalação opcional e isolada

`requirements.txt` e `requirements-lock.txt` mantêm o conteúdo original do
experimento, sem dependências de áudio. `requirements-audio.txt` fixa a
dependência direta `librosa==0.11.0` e usa o lock original como restrição para
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
áudio. Para a futura comparação numérica, será necessário registrar também
Python, todas as versões instaladas, sistema operacional, decodificador e
parâmetros utilizados, em um registro separado do lock original.

O suporte a MP3 depende do backend disponível no ambiente. Se houver falha
de decodificação, verifique o erro original encadeado; conforme o backend,
pode ser necessário disponibilizar FFmpeg no `PATH`. Os testes unitários
abaixo não verificam a instalação de um decodificador.

## Uso do extrator

```python
from src.audio_features import extract_features

features = extract_features("caminho/para/faixa.mp3")
print(features.shape)  # (518,)
```

O carregamento usa `sr=None` e `mono=True`: preserva a taxa de amostragem e
converte para mono. O arquivo inteiro é analisado, sem corte, normalização de
volume ou seleção automática de trecho. A política para músicas completas
será definida antes da avaliação externa.

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

A validação futura deve comparar os 518 valores de MP3s oficiais, identificados
por `track_id`, com as respectivas linhas do CSV e investigar diferenças por
família de atributos. Correspondência de colunas e testes com mocks não
substituem essa comparação.

## Testes automatizados

Os testes usam `unittest` da biblioteca padrão, NumPy, pandas e SciPy. Não
exigem Librosa, FFmpeg, rede, treinamento ou áudio real. O teste de esquema lê
somente o cabeçalho de `data/fma_small_features.csv`; execute na cópia do
projeto que contém esse arquivo.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Também é possível usar o Python do ambiente `.venv-audio`. A opção `-B` evita
gravar bytecode durante a verificação.

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
