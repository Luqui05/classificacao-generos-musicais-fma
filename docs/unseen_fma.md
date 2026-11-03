# Faixas adicionais da FMA para avaliar a inferência

## Comparar com os atributos oficiais das mesmas faixas

```powershell
.\.venv-audio\Scripts\python.exe -B -m scripts.audio.compare_unseen_official --audio-run artifacts/unseen_evaluation/20261103T005557916627Z --models artifacts/final_models/20261029T190158629599Z
```

A comparação reutiliza os vetores extraídos salvos na execução anterior e
confere que reproduzem exatamente suas 200 predições. Aplica os mesmos modelos
às mesmas 40 linhas de `features.csv`, com as colunas na ordem esperada.
A saída fica em nova pasta de `artifacts/unseen_official_comparison/`, ignorada
pelo Git. `summary.csv` contém métricas e mudanças de acerto;
`paired_predictions.csv` permite conferir cada decisão. Também salva atributos
oficiais selecionados, relatórios por classe, matrizes de confusão, contagens
de atributos na tolerância original e hashes de proveniência.

Resultados e interpretação: `validation/unseen_fma/OFFICIAL_COMPARISON.md`.

## Avaliar os cinco pipelines

Após conferir os áudios, executar na raiz do projeto:

```powershell
.\.venv-audio\Scripts\python.exe -B -m scripts.audio.evaluate_unseen_fma --models artifacts/final_models/20261029T190158629599Z --ffmpeg validation/audio_features/runs/tools/imageio_ffmpeg_0_6_0/ffmpeg-win-x86_64-v7.1.exe
```

A saída fica em uma pasta nova de `artifacts/unseen_evaluation/`, ignorada pelo
Git. Cada faixa tem seus atributos e registro de inferência com hashes;
`predictions.csv` contém as 200 predições, `metrics.json` reúne acurácia,
precisão, recall e F1, e os CSVs de confusão usam gêneros verdadeiros nas linhas
e preditos nas colunas. `run.json` registra execução completa ou falha.
O comando verifica sobreposição com os IDs dos modelos antes de iniciar.

Seleção inicial: 40 faixas, cinco por cada um dos oito gêneros do experimento.
O manifesto está em `validation/unseen_fma/sample/manifest.csv`, com IDs,
artistas, títulos, licença original e caminhos dos MP3s. `selection.json`
registra os critérios, contagens e hashes dos arquivos de entrada e do script.

São faixas externas ao conjunto de treinamento, provenientes da mesma FMA.
Isso permite avaliar a inferência sobre faixas não utilizadas no treinamento,
mas não mede generalização para uma base independente. Com cinco faixas por
classe, os resultados devem ser apresentados como avaliação exploratória.

## Critério congelado antes das predições

- Considerar somente `set/subset == medium` e `track/genre_top` pertencente às
  oito classes do experimento. A Small inteira fica excluída, inclusive as
  faixas removidas na deduplicação do treinamento.
- Exigir ID de artista conhecido, título e 518 valores oficiais finitos.
- Excluir igualdade numérica exata dos 518 atributos com qualquer faixa da
  Small que possua atributos válidos. A comparação normaliza o zero negativo.
- Para cada gênero em ordem alfabética, priorizar artistas ausentes da Small
  e ordenar por SHA-256 do texto UTF-8 `42:genero:track_id`, com ID como desempate.
- Selecionar cinco faixas por gênero, sem repetir artistas nem vetores de
  atributos entre as selecionadas. Artistas da Small podem ser necessários:
  `artist_seen_in_small` torna esses casos explícitos. Exigir artistas sempre
  inéditos não permitiu completar International.

Igualdade de atributos é uma verificação de redundância, não uma impressão
digital do áudio. IDs diferentes de artistas também não eliminam aliases ou
colaborações. Os arquivos ainda precisam ser decodificados e conferidos.
Falhas de áudio devem ser registradas; eventuais substituições devem ser
definidas antes da avaliação e nunca pelo acerto ou erro dos modelos.

## Reprodução da seleção

Na raiz do projeto, sem necessidade de ativar o ambiente:

```powershell
.\.venv-audio\Scripts\python.exe -B -m scripts.audio.select_unseen_fma --output validation/unseen_fma/runs/reproducao
```

O script usa somente a biblioteca padrão e recusa saída existente. A semente
padrão é 42 e a quantidade é cinco por gênero. Os vetores oficiais são usados
somente para excluir redundâncias; a futura inferência extrairá os atributos
dos MP3s. Nenhum modelo é carregado durante a seleção.

## MP3s necessários

Usar os 40 caminhos de `validation/unseen_fma/sample/mp3_paths.txt`.
Os arquivos pertencem ao pacote oficial
[fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip), com cerca
de 22 GiB e trechos de 30 segundos, conforme o
[repositório oficial](https://github.com/mdeff/fma#data).
Não são necessários os arquivos das outras faixas após a extração seletiva.

O pacote Small não contém esses arquivos. Não confundir duração da gravação
original no metadado com a duração do trecho no ZIP. Os trechos de 30 segundos
não necessariamente reproduzem os atributos oficiais calculados de versões
completas; esta etapa avalia inferência sobre o áudio efetivamente fornecido.

Na raiz do projeto, para baixar o pacote (requer espaço disponível):

```powershell
curl.exe -L --fail --continue-at - --output data/fma_medium.zip https://os.unil.cloud.switch.ch/fma/fma_medium.zip
```

Conferir a integridade pelo SHA-1 publicado pela FMA:

```powershell
$expected = 'c67b69ea232021025fca9231fc1c7c1a063ab50b'
$actual = (Get-FileHash data/fma_medium.zip -Algorithm SHA1).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'O ZIP diverge do hash oficial; não extraia.' }
```

Extrair apenas as 40 faixas usando o `tar.exe` do Windows:

```powershell
$members = @(Get-Content validation/unseen_fma/sample/mp3_paths.txt | ForEach-Object { "fma_medium/$_" })
tar.exe -xf data/fma_medium.zip -C data @members
if ($LASTEXITCODE -ne 0) { throw 'Falha na extração; confira a saída do tar.' }
$missing = @(Get-Content validation/unseen_fma/sample/mp3_paths.txt | Where-Object { -not (Test-Path -LiteralPath (Join-Path 'data/fma_medium' $_)) })
if ($missing.Count) { throw "MP3s ausentes: $($missing -join ', ')" }
```

O resultado será `data/fma_medium/NNN/NNNNNN.mp3`. Os áudios e o ZIP ficam
ignorados pelo Git. Não é necessário movê-los para a pasta do teste manual.
Esta preparação não executa treinamento nem inferência e não altera os
artefatos ou resultados anteriores.
