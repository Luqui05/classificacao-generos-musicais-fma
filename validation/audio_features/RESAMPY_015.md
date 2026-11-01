# CQT, CENS e tonnetz: investigação das 57 divergências de 006406

## Hipóteses e evidências — 29/10/2026

1. **Afinação automática:** execução das funções de pitch da Librosa 0.5.0
   com STFT histórica produziu -0,06, igual ao estimador atual. O resultado
   permaneceu 461/518. Essa hipótese não explicou as diferenças observadas.
2. **Interpolador:** o adaptador já usava coeficientes Kaiser de resampy 0.1.5,
   mas executava o interpolador de resampy 0.4.3. Comparar as implementações
   revelou uma diferença no limite do laço da asa direita do filtro.
3. **Teste isolado:** substituir somente o interpolador por uma transcrição
   mono do algoritmo antigo elevou 006406 de 461 para **518/518**.
   A verificação de regressão em 030520 também produziu **518/518**.

No Cython 0.1.5, o limite da asa direita usa
`min(n_orig - i - n + 1, (nwin - offset) // index_step)`, onde `i` vem do
laço anterior da asa esquerda. A implementação atual usa o número de amostras
restantes à direita, `n_orig - n - 1`. O comportamento antigo pode truncar
a contribuição à direita perto do fim do sinal. Ele é reproduzido aqui como
comportamento histórico, não recomendado como novo algoritmo de reamostragem.

Uma diferença na CQT altera seu chroma. CENS normaliza, quantiza e suaviza
esse chroma; tonnetz é calculado a partir de CENS no extrator. Assim, o efeito
se propaga às três famílias. A mudança isolada resolveu as 15 diferenças em
chroma CQT, 22 em CENS e 20 em tonnetz.

## Reprodução e limites

Script preservado: `scripts/audio/probe_resampy_015.py`.
Executar da raiz, com as ferramentas locais e o ambiente de áudio já preparados:

```powershell
.\.venv-audio\Scripts\python.exe -B -m scripts.audio.probe_resampy_015 --track-id 6406 --audio validation/audio_features/runs/full_006406_20261029T181918666651Z/006406_full.mp3
```

O script transpõe manualmente o laço Cython para Numba, em mono, acrescentando
uma verificação de limite de índice. Mantém coeficientes, precisão da saída,
acumulação e incremento temporal históricos. Não instala resampy antigo e
não altera o extrator principal. A transcrição foi verificada nestas duas
faixas; não estabelece equivalência geral para todas as taxas ou formas de áudio.

Fonte: `resampy-0.1.5/resampy/interp.pyx` do pacote oficial
https://files.pythonhosted.org/packages/e4/6b/7fd07323225d38f4ce056e3f49fbd92ca36c7f1e82f1285cb2fc9b951dde/resampy-0.1.5.tar.gz
(arquivo SHA256 `7225c928bd730fb4e7b03d4c8df9e516093159f368dd2e8395ff4aa5e00fe2e2`).
O hash do pacote está registrado em `diagnostic_sources.json`.

Resultados locais novos, mantendo todos os anteriores:

- Afinação histórica: `runs/full_features_06406_20261029T182950522875Z` — 461/518.
- Interpolador histórico: `runs/full_features_06406_20261029T183117540420Z` — 518/518.
- Regressão 030520: `runs/full_features_030520_20261029T183202338860Z` — 518/518.
- Reprodução com o script preservado:
  `runs/full_features_06406_20261029T183243935217Z` — 518/518.

Nos diagnósticos, `legacy_cens` continua sendo o nome do perfil base. A execução
nova substitui o interpolador externamente. `diagnostic.json` identifica o
arquivo original do diagnóstico; `interpolator_override.json`, na última
execução, registra o hash da transcrição e da fonte adaptada em memória.

As tolerâncias continuam `atol=1e-6`, `rtol=1e-4`. Nenhum atributo foi removido,
nenhum ganho foi ajustado e não houve treinamento ou inferência. A evidência
agora abrange duas faixas completas. O próximo passo de engenharia é decidir
como integrar e testar essa compatibilidade no extrator antes da inferência.
