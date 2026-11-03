# Inferência de um MP3

Para executar os exemplos, defina `$models`, `$audioRun` e `$ffmpeg` conforme o [roteiro completo do README](../README.md#6-reprodução-da-extensão-de-áudio). Os identificadores históricos citados no texto não existem em um clone novo.

Execute da raiz, no ambiente de áudio, usando os pipelines finais já treinados:

```powershell
.\.venv-audio\Scripts\python.exe -B -m scripts.audio.predict_mp3 "C:\musicas\faixa.mp3" --models $models --ffmpeg $ffmpeg
```

Substitua o caminho do MP3 e, se necessário, o do FFmpeg. Para configuração
dos ambientes e recursos históricos, consulte [extração](audio_features.md)
e [integração histórica](../validation/audio_features/INTEGRATION.md).
O executável FFmpeg do exemplo é um artefato local, não distribuído no Git.

O comando processa o MP3 inteiro com `profile='fma-2017'`, extrai as 518
características uma vez e entrega a mesma linha, na ordem esperada, aos cinco
pipelines completos. Cada pipeline aplica seu próprio pré-processamento já
ajustado. Não há novo ajuste ou seleção de hiperparâmetros durante a inferência.

São verificados o estado completo do treinamento, as versões do manifesto,
os hashes dos cinco modelos, as classes e os nomes e a ordem dos atributos.
Use apenas modelos e manifestos de origem confiável: joblib executa código
durante o carregamento, e hashes verificam integridade, não autenticidade.

A saída mostra um gênero por modelo. Não são calculadas probabilidades ou
uma votação final. Os modelos sempre escolhem entre as oito classes conhecidas;
essa escolha não comprova que uma música pertença a um desses gêneros.

Uma pasta nova em `artifacts/inference/` recebe `features.csv` e `prediction.json`,
com predições, perfil, versões, avisos e hashes do áudio, modelos e FFmpeg.
Pode-se escolher uma pasta com `--output`; pastas existentes são rejeitadas.
Os resultados locais são ignorados pelo Git. O código reutilizável está em
`src/inference.py` e o comando em `scripts/audio/predict_mp3.py`.

## Verificação inicial

O teste completo usou a faixa FMA 006406 e produziu Folk nos cinco pipelines.
A execução está em `artifacts/inference/20261101T181832888204Z/`.
Essa faixa pertence à base de treinamento: o resultado valida a integração,
não a generalização. Uma avaliação de músicas externas exige faixas independentes
rotuladas e um protocolo próprio. A compatibilidade numérica do extrator foi
verificada anteriormente em duas faixas completas, conforme os relatórios.

## Execuções posteriores e alcance

Foi executada uma demonstração com um MP3 fornecido pelo usuário, registrada
em `artifacts/inference/20261101T182655665270Z/`, com as cinco predições e
hashes de proveniência. Sem rótulo de referência verificado, essa demonstração
não mede acerto nem superioridade dos modelos.

A avaliação complementar de 40 faixas da FMA Medium excluiu todos os IDs da
Small, selecionou cinco faixas por classe e comparou as 200 predições aos
rótulos oficiais. Quatro identificadores de artistas também estavam presentes
no treinamento. São faixas fora do treinamento, não uma base independente.
O SVM obteve F1 macro de 0,4232 nos atributos extraídos dos trechos de 30
segundos e de 0,4973 nas linhas oficiais das mesmas faixas, sem retreinamento.
As diferenças não isolam o efeito da duração, da decodificação ou do extrator.
Consulte [seleção e reprodução](unseen_fma.md) e
[resultados e limitações](../validation/unseen_fma/OFFICIAL_COMPARISON.md).
