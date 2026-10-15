# Classificação de gêneros musicais com classificadores clássicos

Experimento do trabalho **“Análise comparativa de classificadores supervisionados clássicos para classificação automática de gêneros musicais”**, desenvolvido no curso de Engenharia de Software do Instituto Federal do Paraná (IFPR), Campus Paranavaí.

O projeto compara **Perceptron, SVM com núcleo RBF, KNN, árvore de decisão e Naive Bayes gaussiano** sobre as mesmas características acústicas e partições de validação da base FMA Small.

**Autor:** Lucas Henrique Teixeira Ribeiro.  
**Coautores do artigo:** Marcelo R. Trierveiler e Ayslan T. Possebom.

## Visão geral

| Item | Configuração |
| --- | --- |
| Base original | FMA Small: 8.000 faixas, oito gêneros e 1.000 faixas por gênero |
| Representação | 518 atributos acústicos previamente extraídos pela FMA |
| Base após remoção de redundâncias | 7.984 faixas |
| Validação | Estratificada, com dez partições, embaralhamento e semente 42 |
| Métrica principal | F1 macro |
| Outras métricas | Acurácia, precisão macro e revocação macro |
| Análise estatística | Friedman e Wilcoxon bilateral, com correção de Holm para dez comparações |
| Visualizações | Comparação de métricas, boxplot, ranks, F1 por gênero e matrizes de confusão |

Os oito gêneros são Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop e Rock.

## Estrutura do projeto

| Caminho | Finalidade |
| --- | --- |
| `prepare_dataset.py` | Prepara o CSV do experimento a partir dos arquivos oficiais |
| `main.py` | Carrega os dados e executa os cinco modelos nas dez partições |
| `run_statistics.py` | Executa os testes estatísticos sobre as métricas salvas |
| `run_visualizations.py` | Gera figuras e tabelas por gênero a partir dos resultados salvos |
| `src/data_loader.py` | Validação dos dados, identificação e remoção de redundâncias |
| `src/models.py` | Configurações dos classificadores e pipelines |
| `src/cross_validation.py` | Validação cruzada, métricas e previsões fora do treinamento |
| `src/statistical_tests.py` | Friedman, ranks, Wilcoxon e Holm |
| `src/visualizations.py` | Construção das figuras e matrizes de confusão |
| `data/fma_metadata/` | Arquivos oficiais `tracks.csv` e `features.csv`, obtidos separadamente |
| `data/fma_small_features.csv` | CSV utilizado pelo experimento |
| `results/tables/` | Diagnósticos dos dados, métricas, previsões e atribuições das partições |
| `results/statistics/` | Resultados estatísticos e relatório textual |
| `results/figures/` | Figuras exportadas em PNG |
| `results/tables/visualization/` | Métricas por gênero e matrizes em CSV |
| `requirements.txt` | Dependências diretas, sem fixação de versões |
| `requirements-lock.txt` | Versões registradas no ambiente do experimento |

## Requisitos

- Python 3.11, conforme o ambiente declarado no artigo.
- `pip` e o módulo `venv` para criar um ambiente isolado.
- Dependências: NumPy, pandas, scikit-learn, SciPy, statsmodels e Matplotlib.
- Acesso aos arquivos oficiais da FMA apenas para preparar novamente os dados.

Os áudios não são necessários: o projeto utiliza atributos já extraídos. Não é necessário instalar Librosa para executar este experimento, nem baixar `fma_small.zip`.

Prefira instalar **`requirements-lock.txt`** para utilizar as versões registradas. O arquivo `requirements.txt` permite instalar as dependências diretas, mas pode selecionar versões diferentes. Versões de Python e bibliotecas devem ser registradas ao reproduzir os resultados.

## 1. Obter o projeto

Baixe o ZIP deste repositório e extraia seu conteúdo, ou clone-o pelo Git. Abra um terminal na pasta que contém `main.py`, `prepare_dataset.py` e os arquivos de dependências.

Os comandos das próximas seções devem ser executados nessa pasta.

## 2. Criar o ambiente e instalar as dependências

### Windows 11 — PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Os comandos utilizam diretamente o Python do ambiente virtual; não é necessário ativá-lo nem alterar a política de execução do PowerShell.

### Linux ou macOS

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
```

## 3. Obter e preparar os dados

### Origem

Os dados são disponibilizados pelo projeto oficial [FMA — A Dataset for Music Analysis](https://github.com/mdeff/fma).

Baixe **[fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)**, indicado na seção [Data do README oficial](https://github.com/mdeff/fma#data), e extraia:

- `tracks.csv`: metadados, incluindo o subconjunto de cada faixa e seu gênero principal;
- `features.csv`: atributos acústicos previamente extraídos com Librosa.

Coloque os dois arquivos em **`data/fma_metadata/`**. O script espera `data/fma_metadata/tracks.csv` e `data/fma_metadata/features.csv`, sem outra pasta intermediária.

### Preparação do CSV

No Windows:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

No Linux ou macOS:

```bash
.venv/bin/python prepare_dataset.py
```

O script seleciona as faixas cujo campo `set/subset` é `small`, obtém o rótulo em `track/genre_top`, alinha os atributos pelo identificador e remove registros sem gênero, caso existam. Os nomes hierárquicos dos atributos são convertidos em nomes simples, como `chroma_cens_kurtosis_01`.

A saída é **`data/fma_small_features.csv`**, com 8.000 linhas e 520 colunas:

- `track_id`: identificador, preservado para rastreabilidade;
- 518 colunas numéricas: entradas dos classificadores;
- `genre`: gênero musical a ser previsto.

O identificador e o gênero não são utilizados como atributos de entrada. A remoção das redundâncias ocorre posteriormente, na execução de `main.py`.

Se o CSV preparado já estiver presente, é possível executar o experimento sem baixar novamente os arquivos oficiais. Execute a preparação para reproduzir essa etapa; ela sobrescreve o CSV de destino.

### Verificação da base preparada

O CSV utilizado no experimento e o CSV regenerado foram comparados e apresentaram igualdade byte a byte, incluindo a ordem das linhas e colunas. O SHA-256 correspondente é:

```text
cc5ae976f8214967747c1260bf9412e6d7900f9eea08102e6fd9423e83b4122d
```

Para conferir no PowerShell:

```powershell
Get-FileHash .\data\fma_small_features.csv -Algorithm SHA256
```

Diferenças de serialização ou de quebras de linha podem alterar o hash; nesse caso, compare também os valores, os nomes das colunas e a ordem dos registros.

## 4. Executar o experimento

Execute os três scripts **nesta ordem**.

### Windows

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe run_statistics.py
.\.venv\Scripts\python.exe run_visualizations.py
```

### Linux ou macOS

```bash
.venv/bin/python main.py
.venv/bin/python run_statistics.py
.venv/bin/python run_visualizations.py
```

| Etapa | Operação | Treina modelos? |
| --- | --- | --- |
| `main.py` | Valida os dados, remove redundâncias e executa 50 treinamentos e avaliações | Sim |
| `run_statistics.py` | Lê `fold_results.csv` e calcula ranks e testes | Não |
| `run_visualizations.py` | Lê métricas, previsões e ranks para gerar tabelas e figuras | Não |

O tempo de execução depende do computador. Os scripts criam as pastas de saída e sobrescrevem os arquivos correspondentes; preserve uma cópia dos resultados anteriores se precisar compará-los.

## 5. Consultar os resultados

| Arquivo | Conteúdo |
| --- | --- |
| `results/tables/dataset_summary.csv` | Quantidades, redundâncias e diagnósticos da base |
| `results/tables/duplicate_samples.csv` | Registros dos grupos redundantes e decisões de manutenção ou remoção |
| `results/tables/class_distribution.csv` | Quantidade final de faixas por gênero |
| `results/tables/fold_results.csv` | Métricas e tempos de cada modelo em cada partição |
| `results/tables/metrics_summary.csv` | Estatísticas descritivas das métricas por modelo |
| `results/tables/predictions.csv` | Previsões de teste de cada faixa e classificador |
| `results/tables/fold_assignments.csv` | Partição de teste atribuída a cada faixa |
| `results/statistics/f1_scores_by_fold.csv` | F1 macro de cada modelo por partição |
| `results/statistics/average_ranks.csv` | Ranks médios e medidas de F1 macro |
| `results/statistics/friedman_result.csv` | Estatística e valor de p de Friedman |
| `results/statistics/posthoc_wilcoxon_holm.csv` | Comparações pareadas, valores de p brutos e ajustados |
| `results/statistics/statistical_report.txt` | Relatório resumido da análise estatística |
| `results/tables/visualization/class_metrics.csv` | Precisão, revocação e F1 por gênero e modelo |

As figuras incluem `class_f1_heatmap.png`, `confusion_matrix_svm.png`, `metrics_comparison.png`, `f1_boxplot.png`, `average_ranks.png` e as matrizes dos demais classificadores. Elas ficam em `results/figures/`.

### Resultados de referência

| Classificador | F1 macro média | Rank médio |
| --- | ---: | ---: |
| SVM | 0,6129 | 1,0 |
| Perceptron | 0,4766 | 2,4 |
| KNN | 0,4710 | 2,6 |
| Árvore de decisão | 0,3660 | 4,0 |
| Naive Bayes gaussiano | 0,2509 | 5,0 |

Friedman apresentou estatística de 38,08 e valor de p de aproximadamente 1,08 × 10⁻⁷. As comparações pareadas foram significativas, após Holm, exceto Perceptron × KNN, com p ajustado de 0,4316. As outras nove comparações apresentaram p ajustado de aproximadamente 0,0195.

Esses valores são referências dos resultados salvos, não uma garantia de identidade entre ambientes. Os tempos de execução também variam com o hardware e a carga do sistema.

## Protocolo e limites de interpretação

- Foram removidos 16 registros excedentes em 12 grupos com os mesmos 518 atributos e o mesmo gênero, mantendo a primeira ocorrência antes da criação das partições.
- Um par com atributos idênticos e gêneros diferentes foi mantido pelo critério adotado. Na divisão utilizada, as duas faixas pertencem à mesma partição de teste.
- A padronização é ajustada somente no treinamento de cada partição, dentro dos pipelines. A árvore de decisão não utiliza essa transformação.
- Todos os classificadores recebem as mesmas partições. Cada faixa tem uma previsão de teste por modelo, chamada previsão *out-of-fold*.
- Os hiperparâmetros são fixos, sem otimização. As conclusões se referem às configurações avaliadas.
- A divisão oficial da FMA não é utilizada, e as partições não são agrupadas por artista. O protocolo não assegura avaliação em artistas ausentes do treinamento.
- Os conjuntos de treinamento das partições se sobrepõem; os resultados estatísticos devem ser interpretados considerando essa dependência.

## Problemas comuns

| Situação | Como resolver |
| --- | --- |
| `tracks.csv` ou `features.csv` não encontrado | Confira se ambos estão diretamente em `data/fma_metadata/` |
| `fma_small_features.csv` não encontrado | Execute `prepare_dataset.py` ou coloque o CSV preparado em `data/` |
| `fold_results.csv` não encontrado | Execute `main.py` antes da análise estatística |
| `average_ranks.csv` não encontrado | Execute `run_statistics.py` antes de gerar as figuras |
| `ModuleNotFoundError` | Instale o arquivo de dependências usando o mesmo Python do ambiente virtual utilizado para executar os scripts |
| Base com estrutura inesperada | Utilize os arquivos oficiais com os cabeçalhos originais; não os reexporte por editores de planilha antes da preparação |

## Referência e atribuição dos dados

Defferrard, M., Benzi, K., Vandergheynst, P. e Bresson, X. (2017). **FMA: A Dataset for Music Analysis.** Proceedings of the 18th International Society for Music Information Retrieval Conference, p. 316–323. [Artigo](https://arxiv.org/abs/1612.01840) · [Repositório oficial](https://github.com/mdeff/fma).

Os atributos acústicos são provenientes da FMA; este projeto seleciona e reorganiza os dados para o experimento. Consulte a seção [Acknowledgments and Licenses](https://github.com/mdeff/fma#acknowledgments-and-licenses) para as condições e os créditos dos materiais originais. Os áudios não são redistribuídos neste projeto.
