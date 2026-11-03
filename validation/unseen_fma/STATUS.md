# Andamento da avaliação em faixas adicionais da FMA

Em 02/11/2026, o usuário executou localmente e apresentou a saída:

- Hash SHA-1 do ZIP correspondente ao publicado pela FMA.
- 40 caminhos do manifesto presentes após a extração seletiva.
- Decodificação FFmpeg: 40 testados, zero falhas.
- Extração com perfil `fma-2017`: 40 faixas, cada uma com 518 atributos
  finitos e aprovada por `validate_features`.

Essas verificações foram executadas pelo usuário no ambiente `.venv-audio`.
Os vetores desse teste não foram persistidos. Isso verifica processamento e
esquema, não equivalência numérica com as linhas oficiais dessas 40 faixas.

Avaliação concluída em 02/11/2026 (horário de Brasília), execução UTC
`20261103T005557916627Z`. O comando terminou com código zero, status
`complete`, 40 faixas e 200 predições. Artefatos locais em
`artifacts/unseen_evaluation/20261103T005557916627Z/`.

| Modelo | Acurácia | F1 macro |
| --- | --- | --- |
| SVM | 0,4250 | 0,4232 |
| KNN | 0,4000 | 0,3688 |
| Naive Bayes | 0,3000 | 0,2762 |
| Decision Tree | 0,2250 | 0,2180 |
| Perceptron | 0,2000 | 0,1657 |

São faixas fora do treinamento, pertencentes à própria FMA. O conjunto possui
cinco faixas por classe e usa trechos de 30 segundos. Esses resultados são
exploratórios; não substituem as métricas de validação cruzada do experimento
original nem comprovam generalização para outra base. Os modelos existentes
foram carregados para predição, sem novo treinamento.
