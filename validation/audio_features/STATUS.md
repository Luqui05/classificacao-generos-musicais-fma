# Andamento da extensão do TCC

Atualizado em 28/10/2026. Esta página acompanha os objetivos macro e registra
as evidências da validação, sem substituir os relatórios de execução.

| Objetivo | Estado | Evidência / condição para avançar |
| --- | --- | --- |
| 1. Validar Librosa e executar os testes | Concluído no ambiente atual | Librosa 0.11.0 processou as 16 faixas; `pip check` sem dependências quebradas e 37 testes passaram na `.venv-audio` |
| 2. Comparar MP3s FMA com o CSV oficial | Comparação executada; compatibilidade numérica não validada | 16 faixas divergentes; investigar diferenças antes do treinamento final |
| 3. Treinar e salvar os cinco pipelines finais | Pendente | Aguardar a conclusão da investigação numérica |
| 4. Implementar inferência de uma faixa externa | Pendente | Depende dos pipelines finais e do contrato validado do extrator |
| 5. Comparar as cinco predições | Pendente | Depende da inferência dos cinco pipelines |
| 6. Avaliar conjunto externo rotulado | Pendente | Definir protocolo e conjunto externo após validar o pipeline |
| 7. Considerar interface web simples | Opcional, posterior | Avaliar depois das etapas de inferência e avaliação externa |

## Execução de referência

- Pasta: `runs/20261028T192711309854Z/` (relatórios locais ignorados pelo Git).
- Comando: `.\.venv-audio\Scripts\python.exe -B compare_audio_features.py compare`.
- Tolerâncias diagnósticas: `rtol=1e-4`, `atol=1e-6`.
- 16 faixas processadas, 518 atributos por faixa: 8.288 comparações.
- 465 valores dentro das tolerâncias (5,61%); 7.823 acima (94,39%).
- Nenhuma faixa ausente ou com erro de extração; todas tiveram status `divergent`.
- Código de saída registrado: `1`, indicando divergência numérica.
- Os hashes do extrator, comparador e manifesto conferem com os arquivos
  atuais na data desta revisão.
- SHA-256 de `run.json`: `bc64ad96dea134444bb82745bc88907460714d734b9bffe692c66af040dc8b7f`.

Os 16 MP3s foram inspecionados como estéreo, 44.100 Hz, com duração aproximada
de 29,98–30,00 segundos. O relatório contém três avisos de depreciação de
módulos da biblioteca padrão na primeira faixa; não houve erro de extração.
O backend efetivamente usado pelo Librosa continua não observado pelo comparador.

## O que os números mostram

As diferenças não se limitam a arredondamento. Exemplos da faixa `030520.mp3`:

| Atributo | Oficial | Extraído |
| --- | --- | --- |
| `rmse_mean_01` | 4,6327242851 | 0,0983576253 |
| `spectral_centroid_mean_01` | 2.119,0834961 | 2.372,2248535 |
| `zcr_mean_01` | 0,0964391604 | 0,1280786991 |

Dos 465 valores dentro das tolerâncias, 366 são máximos de chroma CQT/STFT.
Essas medidas têm normalização que pode produzir máximos iguais a um; essa
concentração não comprova compatibilidade das demais estatísticas.

Os testes confirmam o comportamento do código e do comparador com dados
controlados. A execução real demonstra que a instalação está operacional e
que o extrator atual ainda não reproduz os valores publicados nessa amostra.
Os relatórios não permitem atribuir todas as diferenças a uma causa única.

## Próximo trabalho: investigar a compatibilidade

1. Confirmar a procedência dos MP3s e a correspondência entre o conteúdo/trecho
   analisado e a extração histórica que originou o CSV. Um mesmo ID não basta.
2. Em uma faixa, isolar decodificação, amostras retornadas, taxa, duração e
   tratamento das bordas antes de comparar os descritores.
3. Investigar RMS, STFT, Mel/MFCC e CQT, alterando uma hipótese por vez em
   diagnósticos separados. O histórico do Librosa registra correções de
   normalização de RMS e mudanças de padding, mas isso ainda não explica
   quantitativamente as divergências observadas.
4. Repetir a comparação nas 16 faixas após cada ajuste justificado, preservando
   a execução atual como referência e documentando diferenças remanescentes.

Não aumentar as tolerâncias apenas para obter aprovação. A equivalência ou
uma limitação metodológica precisa ser fundamentada antes de avançar para os
pipelines finais. Esta revisão alterou somente a documentação de acompanhamento.

Fontes para investigação:
[normalização de RMS no Librosa](https://librosa.org/doc/0.11.0/changelog.html#v0-7-1),
[padding no Librosa](https://librosa.org/doc/0.11.0/changelog.html#v0-9-0),
[extrator oficial da FMA](https://github.com/mdeff/fma/blob/master/features.py).
