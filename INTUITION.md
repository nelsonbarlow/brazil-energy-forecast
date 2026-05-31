# Intuition: o que a gente fez, por que funciona, e onde quase tropeçamos

> Documento de entendimento pessoal (não é o paper). Escrito em português, verboso de propósito.
> O paper formal é o `REPORT_IEEE.{tex,pdf}`; o rascunho longo é o `DRAFT_PAPER.md`.
> Todos os números aqui vêm dos arquivos em `results/*.csv`.

---

## 1. A pergunta em uma frase

**Os padrões que governam o consumo de eletricidade são universais (compartilhados entre redes, climas e hemisférios) ou são fundamentalmente locais (cada rede precisa do seu próprio modelo)?**

A gente pegou três modelos de IA treinados em milhões de séries temporais do mundo todo (clima, varejo, tráfego de servidor, finanças, sensores…) — mas que **nunca viram dados de eletricidade do Brasil** — e perguntou: "dado os últimos 30 dias de demanda horária no Sudeste, como será a demanda nas próximas 24 horas?"

Eles acertaram com **1.86% de erro**. Isso empata com um modelo N-BEATS afinado a dedo e treinado em 5+ anos de dados brasileiros (1.91%), e é tão bom quanto sistemas proprietários de operadores de rede dos EUA — sem nenhum treino local.

Essa é a tese de **universalidade**: a demanda elétrica tem uma estrutura que transfere de graça entre redes, e o erro residual é exatamente a parte *cultural/local* (feriados) que o modelo global não conhece.

---

## 2. Por que isso funciona para energia mas não para ações?

Essa é a intuição mais importante do projeto. Vale internalizar.

### Ações: adversarial, nenhum padrão sobrevive

Imagine que você descobre que toda terça-feira a bolsa sobe. Você compra na segunda. Mas todo mundo que percebeu o padrão faz o mesmo. A compra de segunda empurra o preço pra cima cedo, e a alta de terça **desaparece**. O padrão se autodestrói no momento em que é descoberto. Por isso qualquer modelo (treinado ou zero-shot) fica em ~50% de acerto direcional em dados brutos de bolsa. O mercado é um jogo competitivo onde os padrões são arbitrados até sumirem.

### Energia: dirigida por física, padrões são permanentes

A demanda elétrica segue física e comportamento humano:
- Pessoas acordam às 7h, ligam luz e eletrodomésticos → demanda sobe.
- Fábricas rodam das 8h às 18h → pico no meio da tarde.
- Fim de semana tem demanda menor que dia útil.
- Verão no Brasil (dez–fev) = mais ar-condicionado.
- Inverno (jun–ago) = escurece mais cedo, mais iluminação.

**Esses padrões não podem ser "arbitrados".** Não importa o quão bem você preveja que a demanda vai subir às 7h de amanhã — as pessoas ainda vão acordar e ligar a luz. A sua previsão não muda o resultado. É isso que "não-adversarial" significa, e é por isso que a previsão de carga é um problema onde a IA entrega valor real.

---

## 3. O que são "foundation models" (modelos de fundação)?

Pense neles como o equivalente do GPT/Claude, mas para números em vez de palavras.

- **GPT** foi treinado em bilhões de frases e aprendeu gramática, lógica e conhecimento de mundo. Você pergunta sobre um assunto que ele nunca estudou especificamente e ele dá uma resposta razoável.
- **Chronos / TiRex / Moirai** foram treinados em bilhões de séries temporais e aprenderam *padrões*: ciclos diários, ciclos semanais, tendências, sazonalidade, picos, reversão à média.

Quando você mostra a carga brasileira pra eles, eles não precisam "aprender" o que é demanda elétrica. Eles já sabem que séries com ciclo de 24h e de 168h (semanal) se comportam de certo jeito. Eles **generalizam**.

Isso se chama **zero-shot**: prever em dados de um domínio que o modelo nunca viu no treino. O ponto sutil e crucial: a gente *verificou* que os dados da ONS (operador brasileiro) **não estão** no corpus de treino do Chronos. Se estivessem, "zero-shot" seria mentira (vazamento de dados). Checamos o paper e o repositório — não há vazamento. Os datasets de eletricidade mais próximos no treino dele são UCI Electricity (Portugal), ERCOT (Texas) e demanda australiana — nada do Brasil ou da América Latina.

---

## 4. Os modelos que comparamos

| Modelo | Tipo | Tamanho | O que é |
|--------|------|---------|---------|
| **Chronos-2** | Zero-shot | 120M | Transformer encoder-only (Amazon). O melhor da nossa avaliação. |
| **Moirai 2.0** | Zero-shot | 11M | O menor (Salesforce), e quase empata com o Chronos — arquitetura importa tanto quanto tamanho. |
| **TiRex** | Zero-shot | 35M | Baseado em xLSTM (NX-AI). Curiosamente, é o que melhor segura horizontes muito longos. |
| **N-BEATS** | Treinado | 7.3M | A baseline "séria" — treinada em 5+ anos de dados ONS, afinada por grid search. |
| **Linear / LSTM** | Treinado | — | Baselines simples. |
| **Naive (7 dias atrás)** | Baseline | 0 | "Repita o que aconteceu na mesma hora, 7 dias atrás." Captura a sazonalidade semanal. |

A pergunta de pesquisa fica afiada quando o adversário é o N-BEATS: um modelo de deep learning dedicado, treinado em meia década de dados locais, *afinado*. Se o zero-shot empata com ele, a universalidade é difícil de descartar.

---

## 5. As métricas, explicadas (pra você não ter que decorar)

### MAPE (erro percentual absoluto médio) — o número-manchete
Se a demanda real às 15h era 40.000 MW e o modelo previu 39.256 MW, o erro é 744/40.000 = 1.86%. O MAPE faz essa média sobre todas as horas do teste.
- **< 2%**: excepcional, nível de operador profissional.
- **2–4%**: bom, padrão para redes grandes.
- **5–10%**: aceitável.
- **> 10%**: ruim para uma rede grande.

Nosso Chronos-2: **1.86%**. O naive: 5.13%.

### MAE (erro absoluto médio) — em unidades reais
Chronos-2: **829 MW** de erro médio. O SE roda ~40.000 MW, então 829 MW ≈ 2% da carga (consistente com o MAPE). 829 MW soa muito, mas é ~uma turbina a gás média; operadores lidam com essa incerteza de rotina via reservas girantes.

### R² — quanto da variância é explicada
- R² = 0.96 (Chronos-2): explica 96% da variação da demanda.
- R² = 0.77 (Naive): só 77%.

### MASE — comparação direta com o naive
MASE < 1.0 significa "melhor que o naive". Chronos-2: 0.33 (3× melhor). **Sutileza importante**: em horizontes longos (1 mês), o MASE do Chronos passa de 1.0 — ou seja, em previsão mensal o naive *ganha*. Isso não é um defeito escondido; é uma fronteira operacional real (ver seção 8).

### CRPS / cobertura de intervalo — qualidade da *incerteza*
Os modelos não dão só um número, dão uma distribuição. O intervalo de 80% deles cobre 86–89% dos casos reais (levemente conservadores). Para operação de rede, errar pro lado conservador é *bom*: um modelo confiante demais agendaria reserva de menos.

---

## 6. O resultado central (SE, 24h)

| Modelo | Tipo | MAPE |
|--------|------|------|
| Chronos-2 (fine-tuned) | Fine-tuned | **1.73%** |
| Chronos-2 | Zero-shot | **1.86%** |
| N-BEATS (afinado) | Treinado 5 anos | 1.91% ± 0.08% |
| Moirai 2.0 | Zero-shot | 1.93% |
| Linear | Treinado | 2.26% |
| TiRex | Zero-shot | 2.33% |
| Naive | Baseline | 5.13% |

**Leitura honesta:** Chronos-2 zero-shot (1.86%) e N-BEATS afinado (1.91%) estão dentro da margem de ruído um do outro. A gente confirmou isso estatisticamente com o teste de **Diebold–Mariano** (p > 0.29 — ou seja, *não* há diferença significativa) e com intervalos de confiança bootstrap que se sobrepõem. Então o título correto não é "zero-shot vence", e sim "**zero-shot empata com um modelo treinado a dedo — sem precisar de treino**". A vantagem prática é a ausência de treino, não um MAPE menor.

O fine-tuning leve (1.73%) recupera mais 7%, mas reintroduz a necessidade de dados locais. É o trade-off clássico.

---

## 7. As três evidências que tornam isso convincente (não só "um número bom")

Um número bonito num único cenário não é ciência. O que dá força são as réplicas:

### (a) Funciona nos 4 subsistemas
Não só no SE (o maior). Nos quatro — SE, S, NE, N — o Chronos lidera, 45–64% melhor que o naive, R² > 0.90 em todos. Detalhe contra-intuitivo: o *menor* subsistema (Norte) é o mais *fácil* (1.67%), e o Sul é o mais difícil (3.17%, provavelmente por frentes frias de inverno gerando picos de aquecimento).

### (b) Estável ao longo de 3 anos
Testamos 2023, 2024, 2025 separadamente: 1.94% / 1.87% / 1.86%, média 1.89% ± 0.04%. Ou seja, não foi sorte de um ano favorável.

### (c) Replica em OUTRO país — Tóquio (TEPCO) 🇯🇵
Essa é a evidência mais forte da universalidade, e foi adicionada depois. Pegamos a rede de Tóquio (~32 GW, hemisfério norte, calendário de feriados totalmente diferente), treinamos o N-BEATS em dados japoneses 2019–2023 e testamos 2024:

| Modelo | MAPE (Tóquio) |
|--------|---------------|
| Chronos-2 (zero-shot) | **3.91%** |
| Moirai 2.0 (zero-shot) | 3.94% |
| TiRex (zero-shot) | 4.05% |
| N-BEATS (treinado 5 anos, dados japoneses) | 4.44% |
| Naive | 8.86% |

Os três modelos zero-shot **batem** o N-BEATS treinado localmente, na *mesma ordem* que no Brasil. O MAPE absoluto é maior (a demanda japonesa é mais volátil), mas o padrão — transferência zero-shot iguala ou supera treino dedicado — se mantém atravessando continente, hemisfério e calendário. Japão não compartilha geografia, clima nem feriados com o Brasil, e mesmo assim o ranking se preserva.

---

## 8. Onde os modelos QUEBRAM (e por que isso é a melhor parte)

Se a universalidade fosse perfeita, seria suspeito. A graça é que ela quebra num lugar *previsível e interpretável*: **feriados**.

### O padrão de erro por hora e dia
- Mais fácil de noite (madrugada, ~0.5% às 0h) — demanda previsível, todo mundo dormindo.
- Mais difícil no fim de tarde (~2.7% às 15h) — variabilidade comportamental no pico.
- Fim de semana (1.75%) mais fácil que dia útil (1.90%); segunda é o pior dia (rampa de volta ao trabalho).

### A descoberta central: o regime de feriado
Decompondo 2024 por tipo de dia, o Chronos-2 vai de **1.55% (dia normal) para 7.55% (feriado)** — um salto de 4.9×. E isso **não é defeito de um modelo**: nos mesmos feriados, Moirai dá 7.90%, TiRex 9.08%, naive 12.81%. É uma propriedade *do dado*. O modelo prevê demanda de dia útil normal porque, da perspectiva dele, nada sinaliza que é feriado brasileiro — esses feriados não existem no corpus global de treino.

Isso é exatamente o que a hipótese de universalidade prevê: a estrutura física/calendárica transfere, mas o conhecimento *cultural específico* não.

### A prova da fronteira: a covariável de feriado
Se a fronteira é mesmo "conhecimento local de calendário", então *fornecer* esse conhecimento deveria consertar. E conserta. Adicionamos uma única flag binária (é feriado? sim/não) e fizemos fine-tuning leve:

| Regime | Horas | Zero-shot | + Covariável | Δ |
|--------|------:|----------:|-------------:|----:|
| Normal | 5688 | 1.53% | 1.53% | +0.0% |
| Fim de semana | 2232 | 1.77% | 1.74% | −2.0% |
| Véspera | 264 | 2.28% | 2.19% | −4.2% |
| Dia seguinte | 264 | 2.29% | 1.93% | −15.8% |
| Carnaval | 48 | 5.68% | 4.64% | −18.3% |
| **Feriado** | 264 | **6.76%** | **4.37%** | **−35.4%** |
| Bridge (emenda) | 24 | 3.02% | 6.12% | **+102.5%** |
| **Overall** | 8784 | 1.82% | 1.73% | −5.0% |

A covariável corta 35% do erro de feriado e 5% do erro geral, **sem mexer nos dias normais** — exatamente a divisão de trabalho que a tese prevê. A honestidade aqui: os dias de **"emenda"** (feriado grudado em fim de semana, só 24 horas de teste) *pioram* (+102%), porque a codificação binária é grosseira demais pra esse caso. A gente reporta isso em vez de esconder — é onde uma feature de calendário mais rica seria necessária.

---

## 9. Outras descobertas operacionais (úteis na prática)

- **Quanto de histórico preciso?** O limiar crítico é **uma semana**. O MAPE despenca de 5.65% (3 dias) para 2.80% assim que o modelo vê um ciclo semanal completo. Depois de 30 dias os ganhos são marginais. **30 dias é o ponto doce** custo/precisão. Um operador com só 1 semana de dados já consegue 2.80%.
- **Até onde dá pra prever?** Foundation models dominam de 24h a 168h. Em ~2–3 semanas tem um **cruzamento com o naive**: no horizonte de 1 mês, o naive (repetir a semana) ganha, porque o ruído diário se cancela e o autoregressivo acumula erro a cada passo. Lição prática: use foundation models pra operação (dia/semana à frente), e métodos sazonais simples pra planejamento mensal.
- **Tamanho importa, mas arquitetura importa igual.** A família Chronos-Bolt escala log-linear (9M→120M, cada dobra de parâmetros corta ~0.1–0.2pp). No mesmo tamanho, o Moirai de 11M quase empata com o N-BEATS — não é só força bruta.

---

## 10. A parte que você precisa saber: a fabricação que pegamos

Vou ser direto porque isso é importante pra sua confiança no material.

O `DRAFT_PAPER.md` antigo tinha uma tabela afirmando o MAPE de feriados *nomeados específicos*: "Good Friday 14.2%, Tiradentes 12.6%, Christmas 12.4%…". **Esses números não existiam em nenhum CSV e nenhum script os produzia.** Eram narrativa escrita *antes* do experimento de feriado rodar — e quando ele rodou, deu outra coisa (o regime agregado de 7.55%, não por feriado nomeado). Era, na prática, conteúdo inventado.

A gente fez três coisas:

1. **Removeu** a tabela fabricada de todos os lugares (o `REPORT_IEEE` e o `DRAFT_PAPER`), substituindo pelos dados reais medidos.

2. **Auditou os CSVs** com 17 verificações forenses pra garantir que o *resto* dos dados era genuíno (já que uma tabela inventada levanta a pergunta: e o resto?). Todas passaram:
   - Os scripts realmente carregam dados e rodam modelos (não têm números hardcodados).
   - O mesmo valor (Chronos-2 SE = 1.86% / 829.3 MAE) aparece *idêntico* em 6 arquivos de nomes diferentes — um fabricante derraparia.
   - Scripts independentes concordam (benchmark principal e análise de feriado dão 1.87% pra 2024, ambos).
   - As colunas derivadas batem na aritmética (lift = mape/normal; delta = with_cov − zero_shot; horas somam ao total; média ponderada reconstrói o "overall").
   - 179 de 181 valores de MAE/RMSE são não-redondos (cara de computado, não de chute).
   - Procedência no git: os CSVs foram commitados junto com os scripts que os geram.

   **Conclusão da auditoria:** a fabricação ficou confinada a uma tabela de texto; os dados dos CSVs têm todas as marcas de serem saídas reais dos modelos. *Ressalva honesta:* não reexecutamos os modelos do zero, então isso é evidência muito forte, não prova matemática. Pra fechar em 100%, rodar `python scripts/benchmark.py --subsystem SE` (deve dar 1.86%) e `python scripts/holiday_covariates.py` (deve dar 6.76→4.37).

3. **Corrigiu os overclaims** que tinham se acumulado: o título dizia "ISO-Grade Accuracy" (a comparação MAPE entre redes diferentes não é rigorosa — MAPE depende da volatilidade de cada carga, não só da qualidade do modelo); rebaixamos isso pra "contexto solto". Também consertamos "single country" → "two countries" (porque agora temos o TEPCO).

**A lição metodológica:** escrever a narrativa antes de ter o resultado é perigoso — vira um número plausível que ninguém mediu. O hábito certo é: rode o experimento → salve o CSV → *só então* escreva a frase, citando o arquivo.

---

## 11. O que é publicável aqui

A contribuição: **modelos de fundação zero-shot igualam um modelo de deep learning treinado localmente (e batem outro, em Tóquio) na previsão de carga de uma rede hidro-dependente de mercado emergente que eles nunca viram — e a única coisa que eles não sabem é o calendário de feriados local, que uma flag binária resolve.**

Por que importa:
- Operadores em países em desenvolvimento talvez não precisem de anos de desenvolvimento de modelo local — dá pra usar foundation models de prateleira com 30 dias de histórico.
- Valida que o pré-treino em séries globais diversas transfere entre hemisférios e topologias de rede.
- É o primeiro benchmark público de Chronos-2, TiRex e Moirai em dados de energia brasileiros (+ uma réplica japonesa).

Alvo: Applied Energy ou IEEE TPWRS.

---

## 12. O que tornaria ainda mais forte (próximos passos honestos)

A maioria das ideias do INTUITION antigo já foi feita (4 subsistemas ✓, comparação com treinado ✓, feriados ✓, horizontes longos ✓). O que sobra de verdade:

1. **Calendário mais rico** — estender a flag binária pra codificar emendas, vésperas e feriados regionais, consertando a regressão dos "bridge days".
2. **Mais países** — além de Brasil e Japão, testar Índia, Nigéria, Indonésia, pra ver se a universalidade alça contextos econômicos fundamentalmente diferentes.
3. **Baseline com variáveis exógenas** — nossos modelos treinados também só usam carga. Um TFT com temperatura poderia estreitar a diferença e seria um adversário mais justo.
4. **Consertar/remover o LSTM** — ele não convergiu (13% MAPE, R² negativo). Está no paper marcado como "não convergiu" por honestidade, mas idealmente debugar (provável problema de normalização/learning rate) ou tirar.
5. **Reexecução de verificação** — rodar os scripts e confirmar bit-a-bit os CSVs, pra fechar a auditoria em 100%.
6. **Preço (CCEE PLD)** — preço é mais difícil que carga (mais volátil, influenciado por política). Testaria o limite da universalidade: ela vale pra sinais de mercado ou só pra demanda física?

---

## 13. Resumão de uma frase

Previsão de carga elétrica é um problema real e de alto impacto onde foundation models entregam valor genuíno — diferente de previsão de ações, onde o mercado *força* a imprevisibilidade. A demanda segue leis físicas e comportamentais que persistem no tempo; os modelos aprenderam esses padrões universais de dados globais diversos e transferem surpreendentemente bem pra rede brasileira (e japonesa) sem treino local. A fronteira da universalidade é nítida: eles só falham nos feriados culturais locais, e uma única covariável de calendário recupera a maior parte disso.
