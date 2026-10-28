# Comparação numérica com a FMA — etapa 3

O comando `compare_audio_features.py` compara o extrator atual com as linhas
do arquivo oficial `data/fma_metadata/features.csv`. Os modelos e os resultados
da validação cruzada não participam dessa etapa.

**Estado atual:** comparação com áudio real executada em 28/10/2026, nas 16
faixas selecionadas. Todas apresentaram divergências com as tolerâncias
diagnósticas atuais; a compatibilidade numérica continua não validada.
Consulte o [andamento e a análise da execução](validation/audio_features/STATUS.md).

## MP3s necessários

Disponibilize os **MP3s originais do arquivo `fma_small.zip`**, sem converter,
recortar, normalizar volume ou modificar tags. O pacote oficial contém trechos
de aproximadamente 30 segundos; uma versão completa da música ou um download
atual da mesma faixa não comprova que os bytes ou o trecho sejam os mesmos.

O pacote é disponibilizado na seção
[Data do repositório oficial](https://github.com/mdeff/fma#data)
([download de fma_small.zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip),
aproximadamente 7,2 GiB). Se já possuir o ZIP, pode extrair apenas os 16 membros
listados. Não é necessário disponibilizar todas as 8.000 faixas ao projeto.

A lista foi gerada a partir das 7.984 amostras sem redundâncias, com duas faixas
por gênero, semente 42 e exclusão dos seis arquivos curtos/vazios registrados
na [errata oficial](https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata).
É uma amostra inicial para diagnóstico; não demonstra equivalência em toda a base.

| Gênero | Arquivos relativos a `data/fma_small/` |
| --- | --- |
| Electronic | `030/030520.mp3`, `120/120468.mp3` |
| Experimental | `064/064078.mp3`, `064/064252.mp3` |
| Folk | `006/006406.mp3`, `113/113167.mp3` |
| Hip-Hop | `012/012394.mp3`, `089/089860.mp3` |
| Instrumental | `130/130931.mp3`, `131/131912.mp3` |
| International | `059/059684.mp3`, `113/113271.mp3` |
| Pop | `067/067368.mp3`, `129/129051.mp3` |
| Rock | `040/040133.mp3`, `072/072287.mp3` |

Exemplo de caminho completo no projeto:
`C:\IFPR\tcc\experimento_tcc\data\fma_small\030\030520.mp3`.
Mantenha os seis dígitos dos nomes e a estrutura de pastas de três dígitos.
Os MP3s e os relatórios de execução estão ignorados pelo Git.

A seleção está em `validation/audio_features/sample/manifest.csv`. O arquivo
`selection.json` registra o método, a semente, as exclusões e os hashes dos
CSVs de origem e do manifesto. Os valores das 16 linhas do CSV do experimento
foram conferidos contra o arquivo oficial, permitindo apenas diferenças de
serialização dos CSVs (`rtol=1e-12`, `atol=1e-12`). Essa verificação não envolve áudio.

## Executar quando os áudios estiverem disponíveis

Use o ambiente separado descrito em [audio_features.md](audio_features.md).
Execute na raiz do projeto:

```powershell
.\.venv-audio\Scripts\python.exe -B compare_audio_features.py compare
```

Se os arquivos estiverem em outra pasta, aponte para a raiz que contém `030/`,
`120/` e as demais subpastas:

```powershell
.\.venv-audio\Scripts\python.exe -B compare_audio_features.py compare --audio-root "D:\datasets\fma_small"
```

Cada execução cria uma pasta nova em `validation/audio_features/runs/`, com
data/hora UTC no nome. `--output CAMINHO` permite escolher outra pasta nova;
pastas já existentes são rejeitadas para evitar sobrescritas.

O comando tenta todas as faixas do manifesto. Se apenas parte estiver
disponível, compara essa parte e registra as demais como ausentes. Uma falha
de extração é registrada e não impede as tentativas seguintes.

## Relatórios e interpretação

| Arquivo | Conteúdo |
| --- | --- |
| `tracks.csv` | Status e quantidade de atributos dentro da tolerância por faixa; erros e ausências |
| `extracted_features.csv` | Vetores obtidos nas extrações bem-sucedidas |
| `differences.csv` | Valores oficial/extraído, erros absoluto e relativo e resultado por atributo |
| `family_summary.csv` | Quantidades, erro absoluto médio/máximo e erro relativo máximo por família |
| `family_statistic_summary.csv` | Mesmo resumo por família e estatística, evitando misturar escalas distintas |
| `run.json` | Comando, versões dos pacotes, Python, sistema, hashes, tolerâncias, metadados e avisos dos áudios |

Os três arquivos de diferenças/resumos só são gerados se pelo menos uma faixa
for comparada. Sem áudio, não há números de comparação a interpretar.

Para cada atributo, o critério diagnóstico é:

```text
erro_absoluto = abs(extraído - oficial)
erro_relativo = erro_absoluto / abs(oficial), quando oficial != 0
dentro_da_tolerância = erro_absoluto <= atol + rtol * abs(oficial)
```

Quando o valor oficial é zero, o erro relativo fica vazio no CSV e a comparação
usa a tolerância absoluta. Isso vale também para zero contra zero, que passa
no critério absoluto. Nenhum erro é mascarado com divisão por um epsilon.

Os valores iniciais são `rtol=1e-4` e `atol=1e-6`. São pontos de partida
exploratórios para localizar divergências, sem justificativa experimental
para declará-los limites de equivalência. Para outro diagnóstico, forneça
`--rtol` e `--atol`; registre e justifique a escolha antes de uma conclusão.
Não aumente os limites apenas para obter aprovação.

O código de saída é:

- `0`: todas as faixas solicitadas foram comparadas e todos os atributos estão
  dentro das tolerâncias escolhidas; resultado restrito à amostra e ao ambiente.
- `1`: execução completa, com diferenças acima das tolerâncias.
- `2`: execução incompleta (arquivos ausentes/falhas) ou erro de entrada.
  Erros de entrada são exibidos no terminal; podem ocorrer antes da criação do relatório.

Os CSVs e o manifesto são verificados por SHA-256. Se a referência oficial ou
o manifesto mudar, a comparação exige uma nova seleção. Cada MP3 disponível
também recebe um hash, mas isso registra os bytes recebidos: não autentica a
origem no ZIP oficial.

Os metadados de taxa, canais, frames e duração são consultados com SoundFile,
quando disponível. A extração continua utilizando o `librosa.load` original,
com escolha automática do backend. **O backend efetivamente usado pelo
Librosa não é observado pelo comparador**; essa limitação fica explícita em
`run.json`. As versões dos pacotes, do libsndfile quando consultável, e os
avisos de carregamento são registrados para apoiar a investigação.

Diferenças precisam ser investigadas por família/estatística, considerando
decodificação, conteúdo/trecho do áudio, padrões de STFT, CQT, RMS e filtros
Mel. O mesmo `track_id` e as mesmas 518 colunas não provam equivalência dos
valores. Mesmo uma comparação favorável nessa amostra não representa uma
validação em todas as faixas.

## Reproduzir ou ampliar a seleção

O manifesto entregue já está pronto. Para gerar outra amostra, use uma pasta
nova; o comando não substitui a seleção existente:

```powershell
.\.venv-audio\Scripts\python.exe -B compare_audio_features.py prepare --per-genre 2 --seed 42 --output validation/audio_features/another_sample
```

Para compará-la, forneça `--sample validation/audio_features/another_sample`.
A seleção usa o carregador original somente para leitura e remoção das mesmas
redundâncias. Não executa treinamento, avaliação ou gravação em `results/`.

## Verificação automatizada

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Os testes do comparador usam CSVs e valores simulados para verificar cálculo
de erros, tolerâncias, referência zero, leitura do cabeçalho oficial,
integridade dos arquivos de entrada, relatórios, falhas e ausência de áudio.
Eles não validam a extração numérica de MP3s oficiais.
