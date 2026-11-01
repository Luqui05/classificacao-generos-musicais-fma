# Treinamento dos pipelines finais

Execução concluída: `artifacts/final_models/20261029T190158629599Z/`.
Os cinco pipelines foram ajustados e recarregados com equivalência de estado
aprendido verificada campo a campo, sem realizar predições. Os três testes
específicos passaram, incluindo detecção de alteração no estado de uma árvore.
A execução anterior `20261029T185928176239Z` ficou preservada como `failed`:
a comparação inicial por hash não era adequada ao padding interno da árvore.

Execute na raiz do projeto, no ambiente original:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.training.train_final_models
```

O script verifica todas as versões fixadas em `requirements-lock.txt`, reutiliza
`load_dataset(..., drop_duplicate_samples=True)` e `build_models()` e ajusta
os cinco pipelines nos 7.984 registros e 518 atributos oficiais. Não muda
hiperparâmetros nem reexecuta a validação cruzada. O StandardScaler de cada
pipeline que o utiliza é ajustado sobre toda a base destinada ao treinamento
final. A árvore continua sem padronização, conforme a fábrica original.

Cada execução cria um diretório exclusivo em `artifacts/final_models/`, com
cinco arquivos joblib e `manifest.json`. É possível informar `--output`, mas
um diretório existente é rejeitado para evitar sobrescrita. Falhas ficam
registradas com status `failed`; somente `complete` identifica uma execução
concluída. Os arquivos joblib incluem o pipeline inteiro, com pré-processamento.

O manifesto contém os parâmetros efetivos, versões, ordem das colunas,
classes, IDs de treinamento, distribuição das classes, tempos e avisos,
além dos hashes dos modelos, dataset, lock, fábrica, carregador e resultados
originais. A rotina recarrega apenas os artefatos que acabou de criar e
compara o estado serializado com o modelo em memória, sem chamar predict.
Arquivos joblib devem ser carregados somente de origem confiável e com as
versões registradas, pois a desserialização executa código Python.

Não se calculam métricas sobre o conjunto usado no ajuste final. As métricas
originais de validação cruzada permanecem como estimativas de desempenho;
os modelos finais são novos ajustes sobre a base completa. A avaliação
externa e a inferência de áudio são etapas posteriores.

Versionar este documento, o script, os testes e a regra de exclusão no Git.
Os binários de modelos e os manifestos de execução permanecem em
`artifacts/final_models/`, ignorado pelo Git; preserve essa pasta em backup
para usar os modelos sem precisar repetir o treinamento.
