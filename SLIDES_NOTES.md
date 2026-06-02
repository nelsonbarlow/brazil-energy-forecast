# Roteiro do vídeo (≤ 7 min)

Slides: **`SLIDES.pdf`** (fonte `SLIDES.tex`). As mesmas notas estão embutidas em cada slide
via `\note{}` — para gerar um PDF de notas ao lado dos slides, troque no preâmbulo do `.tex`:
`\setbeameroption{show only notes}` e recompile com outro nome de saída.

**Orçamento total: ~6 min 40 s** (margem de 20 s para os 7 min). Fale em ritmo calmo; os números
são o que importa, deixe-os respirar.

---

### Slide 1 — Título · ~20s
Olá. Meu projeto investiga uma pergunta simples mas de grande impacto prático: será que modelos de
fundação de séries temporais conseguem prever a demanda de energia do Brasil **sem nunca terem sido
treinados em dados brasileiros**? Eu chamo essa hipótese de *universalidade da demanda elétrica*.

### Slide 2 — O problema · ~30s
Hoje, prever carga elétrica significa que cada operador constrói um modelo sob medida, treinado em
anos de dados locais. Isso funciona, mas é caro — e impossível onde não há histórico: uma microrrede
nova, uma região recém-conectada, uma rede em desenvolvimento. A pergunta é: dá pra pular essa etapa?

### Slide 3 — A hipótese · ~40s
A intuição é que a demanda elétrica tem duas camadas. Uma **universal** — física de aquecimento e
refrigeração, ritmo circadiano, o ciclo semanal de trabalho — que produz o mesmo formato de curva em
qualquer lugar do mundo. E uma **local** — feriados, hidrologia, a estação invertida do hemisfério sul.
Minha hipótese: um modelo treinado em séries globais já aprendeu a camada universal. Então ele deveria
acertar zero-shot, e errar só na parte local. Vou testar exatamente isso.

### Slide 4 — Dados e modelos · ~35s
Uso dados horários da ONS, sete anos, os quatro subsistemas do sistema interligado. Comparo três
modelos de fundação zero-shot — Chronos-2, Moirai e TiRex — contra um N-BEATS ajustado e treinado em
cinco anos de dados locais, mais baselines. Importante: verifiquei que os dados da ONS **não** estão
no corpus de treino do Chronos — não há vazamento. Os modelos zero-shot só enxergam a janela de contexto.

### Slide 5 — Resultado principal · ~50s
Este é o resultado central. No subsistema Sudeste, horizonte de 24 horas: o Chronos-2 zero-shot atinge
**1,86%** de MAPE. O N-BEATS, treinado em cinco anos de dados locais, fica em **1,91%**. Ou seja, o
modelo que **nunca viu dados brasileiros** empata com — na verdade, levemente supera — o modelo treinado
localmente. Todos os modelos de fundação batem o baseline naive de 5,13% com folga. Fine-tuning leve
ganha só mais 7%.

### Slide 6 — Empate estatístico · ~30s
Alguém poderia dizer: 1,86 contra 1,91, é ruído. Exatamente — e é esse o ponto. O teste de
Diebold-Mariano dá *p* maior que 0,29, então a diferença **não** é estatisticamente significativa. Os
intervalos de confiança se sobrepõem. E garanti que o N-BEATS estava bem ajustado, com busca em grade
e três sementes. A conclusão honesta: mesma acurácia. A diferença é que um exigiu cinco anos de treino
local e o outro, nenhum.

### Slide 7 — Generalização · ~45s
O resultado não é um acaso do Sudeste. Mantém-se nos quatro subsistemas, com R-quadrado acima de 0,90
em todos. Mantém-se em três anos diferentes de teste. Sobre contexto: o modelo precisa de pelo menos
uma semana de histórico — quando vê um ciclo semanal completo, o erro cai pela metade; 30 dias é o
ponto ideal. Em horizonte, os modelos de fundação ganham até duas semanas; depois disso, o naive volta
a vencer. A ferramenta é para operação de curto prazo.

### Slide 8 — Réplica entre países (Tóquio) · ~40s
A evidência mais forte da universalidade: repeti tudo na rede de Tóquio, a TEPCO. País diferente,
hemisfério diferente, calendário de feriados diferente. E o mesmo padrão aparece: os três modelos
zero-shot batem o N-BEATS treinado em cinco anos de dados japoneses. O MAPE absoluto é maior porque a
demanda japonesa é mais volátil, mas o resultado qualitativo se sustenta entre países. Difícil explicar
isso a não ser que exista mesmo uma estrutura universal na demanda.

### Slide 9 — Onde falha: feriados · ~40s
E onde o modelo falha? Exatamente onde a hipótese previu: **feriados**. Em feriados nacionais, o erro
do Chronos-2 sobe de cerca de 1,6% para quatro ou cinco vezes isso. A razão é intuitiva — nada no corpus
global diz a ele que é feriado no Brasil, então ele prevê um dia útil comum. E não é defeito de um
modelo só: todos degradam igual, inclusive o naive. Isso confirma a fronteira: o que não transfere é o
conhecimento de calendário local.

### Slide 10 — A correção funciona · ~35s
E se a falha é falta de conhecimento de calendário, basta fornecer esse conhecimento. Adicionei **uma**
covariável binária de feriado, com fine-tuning leve. O erro de feriado cai **35%**, o Carnaval 18%, o
dia seguinte ao feriado 16% — e os dias normais ficam intactos. Isso fecha o argumento: a fronteira da
universalidade é conhecimento de calendário, não uma limitação de arquitetura.

### Slide 11 — Conclusão · ~35s
Em resumo: a demanda elétrica tem uma universalidade, mas com fronteira precisa. A estrutura física e
comportamental transfere entre redes, anos e até países. Só o calendário local não transfere — e uma
única covariável o recupera. Para um operador com poucos dados, a receita prática é: comece zero-shot
com 30 dias de contexto, faça fine-tuning se puder, adicione a covariável de feriado, e não passe de
duas semanas de horizonte. Previsão day-ahead precisa não exige mais anos de desenvolvimento local.
Obrigado. Todo o código está no GitHub.

---

**Dicas de gravação:** ensaie uma vez cronometrando — se passar de 7 min, corte detalhes do slide 7
(generalização) e do slide 6, que são os mais "compressíveis". Os slides 5, 8, 9 e 10 são o coração
do argumento; não corte deles.
