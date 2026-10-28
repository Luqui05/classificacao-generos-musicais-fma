# Andamento da extensão do TCC

Atualizado em 28/10/2026. Esta página acompanha os objetivos macro e registra
as evidências da validação, sem substituir os relatórios de execução.

| Objetivo | Estado | Evidência / condição para avançar |
| --- | --- | --- |
| 1. Validar Librosa e executar os testes | Concluído no ambiente atual | Librosa 0.11.0 processou as 16 faixas; `pip check` sem dependências quebradas e 42 testes passaram na `.venv-audio` |
| 2. Comparar MP3s FMA com o CSV oficial | Validação parcial; 518 atributos ainda pendentes | Em 030520, sete momentos de RMS reproduzidos na música completa com PCM16 e fórmula histórica; trechos de 30 s continuam divergentes |
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

## Diagnóstico concluído em 030520: RMS e decodificação

O [relatório de RMS](RMS_030520.md) registra três execuções isoladas. No trecho, a fórmula
espectral da Librosa 0.5.0 reduz o erro da média de RMS de 97,88% para cerca de
3,95%, mas os sete momentos ainda ficam fora da tolerância.

SoundFile e FFmpeg 7.1 retornaram 1.321.967 amostras a 44.100 Hz, com atraso
estimado igual a zero. Em float32, o erro RMS entre as formas de onda foi
1,61e-7. A ZCR média ficou igual nos dois decodificadores. O caminho audioread
com FFmpeg e PCM16 também manteve a divergência em relação ao CSV oficial.

O MP3 local foi comparado byte a byte com o membro `fma_small/030/030520.mp3`
servido pelo ZIP oficial e coincidiu. A consulta parcial não verificou o
checksum do ZIP inteiro e não estabelece qual áudio originou o CSV histórico.

Também foi obtida somente a faixa 030520 do ZIP `fma_full`, com duração de
183,457959 s. Nessa versão, os sete momentos de RMS passam nas tolerâncias
originais ao usar PCM16 e a fórmula histórica. Isso indica fortemente que o RMS
publicado para essa faixa descreve a música completa. A ZCR se aproxima do
CSV, mas ainda apresenta divergências. O resultado não valida as demais famílias
nem os outros IDs.

As evidências adicionais estão em `runs/rms_030520_20261028T203213995428Z/`
e `runs/provenance_030520_20261028T203356483128Z/`. O FFmpeg portátil foi
mantido em `runs/tools/`, com versão e hashes registrados no diagnóstico.
A versão completa e sua procedência estão em `runs/full_030520_20261028T203735208700Z/`;
sua comparação está em `runs/rms_030520_20261028T203800305571Z/`.

## Próximo trabalho: investigar a extração histórica

1. Comparar os demais descritores de 030520 usando a versão completa. A duração
   e a fórmula histórica explicam o RMS nesta faixa; não generalizar para todos
   os atributos e IDs antes da validação.
2. Se necessário, reproduzir versões históricas em ambiente isolado. A comparação
   entre decodificadores atuais já foi feita para 030520, tanto no trecho quanto
   na versão completa. A origem do trecho também foi conferida.
3. Investigar RMS, STFT, Mel/MFCC e CQT, alterando uma hipótese por vez em
   diagnósticos separados. O histórico do Librosa registra correções de
   normalização de RMS e mudanças de padding. O efeito desses dois fatores foi
   medido em 030520; a equivalência das demais famílias continua pendente.
4. Ampliar a comparação para outras faixas após confirmar a duração de referência
   e cada ajuste justificado, preservando as execuções atuais. Não exigir que os
   trechos de 30 s coincidam com descritores que representem músicas completas.

Não aumentar as tolerâncias apenas para obter aprovação. A equivalência ou
uma limitação metodológica precisa ser fundamentada antes de avançar para os
pipelines finais. Os diagnósticos não alteraram o extrator nem os resultados
do experimento original.

Fontes para investigação:
[normalização de RMS no Librosa](https://librosa.org/doc/0.11.0/changelog.html#v0-7-1),
[padding no Librosa](https://librosa.org/doc/0.11.0/changelog.html#v0-9-0),
[extrator oficial da FMA](https://github.com/mdeff/fma/blob/master/features.py).
