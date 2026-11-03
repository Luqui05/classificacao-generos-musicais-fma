# Comparação pareada: atributos oficiais versus extraídos

Execução realizada em 02/11/2026, horário de Brasília. Pasta UTC:
`artifacts/unseen_official_comparison/20261103T010600923526Z/`.

Os cinco pipelines existentes foram apenas carregados, sem ajustar nenhum
parâmetro. Foram usadas as mesmas 40 faixas, os mesmos rótulos e a mesma ordem
de atributos. A reprodução das 200 predições a partir dos vetores extraídos
persistidos foi exata. A comparação terminou com código zero e status complete.

| Modelo | F1 macro extraído | F1 macro oficial | Acertos extraído | Acertos oficial | Predições alteradas |
| --- | --- | --- | --- | --- | --- |
| SVM | 0,4232 | 0,4973 | 17/40 | 20/40 | 21/40 |
| Perceptron | 0,1657 | 0,4062 | 8/40 | 17/40 | 27/40 |
| KNN | 0,3688 | 0,3457 | 16/40 | 16/40 | 22/40 |
| Naive Bayes | 0,2762 | 0,1753 | 12/40 | 10/40 | 26/40 |
| Decision Tree | 0,2180 | 0,1496 | 9/40 | 7/40 | 24/40 |

## Interpretação do SVM

O F1 macro oficial é 0,4972610723, contra 0,4232142857 nos trechos extraídos.
A diferença pareada é 0,0740467865. O uso dos atributos oficiais corrigiu oito
erros, mas converteu cinco acertos em erros; o saldo foi de três acertos.
Das 21 predições alteradas, oito passaram de um gênero errado para outro errado.

Com os atributos oficiais, Folk passou de 3 para 5 acertos e Rock de 2 para 4.
Instrumental passou de zero para um, enquanto Pop continuou sem acertos.
Hip-Hop caiu de cinco para três e Experimental de dois para um.

O F1 macro original de aproximadamente 0,6129 ainda supera 0,4973 por cerca de
0,1156. Não é possível atribuir essa diferença remanescente isoladamente à
generalização: mudam a amostra, o procedimento de seleção e a forma de avaliação
(média dos folds versus uma aplicação dos modelos finais).

Nas 40 faixas, apenas 21 a 51 dos 518 valores por faixa ficaram dentro da
tolerância original (`rtol=1e-4`, `atol=1e-6`). Isso evidencia que os vetores
extraídos dos trechos não são numericamente equivalentes às linhas oficiais.
Não distingue diferenças causadas por trecho/duração, decodificação ou detalhes
do extrator. Também não representa percentual de erro de classificação.

O efeito sobre a classificação depende do modelo: o F1 oficial aumentou no
SVM e no Perceptron, mas diminuiu nos outros três. Portanto, não se deve tratar
o CSV oficial como garantia de aumento da acurácia nem atribuir toda a queda
original do SVM ao processamento de áudio.

## Texto proposto para o artigo

Como diagnóstico complementar, os mesmos pipelines foram aplicados aos
atributos oficiais das 40 faixas selecionadas, mantendo os modelos e rótulos
fixos. O SVM apresentou F1 macro de 0,4973 com os atributos oficiais e de
0,4232 com os atributos extraídos dos trechos de 30 segundos. A mudança de
representação alterou 21 das 40 predições, corrigindo oito erros e introduzindo
cinco novos erros. Esses resultados mostram sensibilidade à origem dos
atributos, mas não isolam o efeito da duração do áudio, da decodificação ou do
extrator. O desempenho com os atributos oficiais permaneceu abaixo da média
da validação cruzada original (aproximadamente 0,6129), em condições de
avaliação distintas. Os resultados devem ser interpretados como exploratórios,
considerando o conjunto reduzido de cinco faixas por gênero.
