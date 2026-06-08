# ZHISA — Концепція візуально-адаптивного трейдинг-AI

> Мультимодальна самонавчальна система, яка торгує фінансові інструменти, спираючись на візуальний аналіз графіків, ринкову мікроструктуру та ринково-структурні патерни. Документ описує виключно концепцію, архітектуру мислення і стратегію навчання — без жодного рядка коду.

---

## 0. Маніфест проекту

Мета — побудувати агента, який:

1. **Бачить** ринок так само, як досвідчений трейдер, що роками читає графіки очима: свічки, обʼєми, рівні, фігури, ринкову структуру, контекст вищих таймфреймів.
2. **Розуміє** контекст: де ми в тренді/діапазоні/розвороті, яка фаза циклу, який режим волатильності, який сентимент.
3. **Памʼятає**: працює не з одним скріншотом, а з потоком станів, здатна порівнювати поточну ситуацію з тим, що вже бачила у схожих режимах.
4. **Діє** як трейдер-практик: вхід, частковий вихід, пересування стопу, ре-ентрі, утримання — а не одне передбачення «вгору/вниз».
5. **Навчається безперервно**: ринок — це не статичний датасет, а нестаціонарне середовище, що еволюціонує. Модель має адаптуватись у реальному часі.
6. **Оцінює себе за правильними метриками**: ризик-скоригований прибуток (Sharpe, Sortino, Calmar), а не точність напрямку.

Фінальна ціль — стабільно позитивне математичне очікування після всіх транзакційних витрат, на рівні, що переверщує benchmark (buy & hold, ринкові індекси) з контрольованим drawdown.

---

## 1. Проблема та чому саме такий підхід

### 1.1. Чому «просто передбачити напрямок» — не працює

Класичні спроби побудувати трейдинг-AI через supervised classification «up/down» мають фундаментальні дефекти:

- **Дисбаланс класів і тонкий edge.** Навіть якщо модель має 55% точності по напрямку — це може бути збитково після комісій, спредів і прослизання.
- **Шумні лейбли.** «Що було б, якби я увійшов на цьому барі?» — це контрфактичне питання, на яке немає прямої відповіді в історичних даних. Відомий «up-bar» міг би стати down-bar за інших умов.
- **Нестаціонарність.** Розподіл прибутковостей змінюється: тренди, флети, кризи, regime-переключення. Модель, натренована на 2017–2021, ламається на 2022+.
- **Відкладена, розріджена винагорода.** Рішення «увійти в лонг» приймається тут і зараз, а PnL реалізується через N барів — класична проблема credit assignment.
- **Хибне відчуття точності.** Accuracy 60% на daily-данних з 7-річного бектесту часто означає, що модель просто вгадує напрямок тренду, а реальне edge — нульове.

### 1.2. Чому візуальний підхід — перспективний

Люди-трейдери десятиліттями заробляли на **візуальних** патернах. Очі:

- знаходять рівні підтримки/опору миттєво;
- розрізняють «хвильову структуру» Елліотта, фігури голова-плечі, прапори, канали;
- оцінюють «якість» свічки (engulfing, hammer, doji у контексті);
- сприймають щільність/розрідженість обʼєму;
- бачать «атмосферу» — зрілість тренду, його виснаження.

Сучасні vision-моделі (CNN, Vision Transformers, мультимодальні foundation models) мають усі технічні можливості, щоб перевершити людину в цій задачі — якщо дати їм правильні дані, правильний лосс і правильний механізм навчання.

### 1.3. Чому чистий RL «з нуля» — теж не працює

Reinforcement learning у трейдингу страждає від:

- **Sample inefficiency.** Мільйони епізодів потрібні навіть для простої стратегії. У нас — лише один-єдичний шлях часу для кожного інструменту.
- **Exploration без інфраструктури.** Випадкові угоди в реальному ринку = збитки.
- **Sparse reward.** PnL приходить рідко, сигнал слабкий.
- **Складного credit assignment.** Яка саме дія призвела до прибутку через 200 барів?

**Висновок:** потрібен гібрид — сильний supervised/self-supervised bootstrap на величезних масивах історичних даних, потім — RL-дошліфовка з реалістичним симулятором виконання.

---

## 2. Доменна модель ринку

### 2.1. Інструменти та класи активів

Старт фокусується на **ліквідних деривативах крипторинку** (perpetual futures на BTC, ETH, SOL та ін. на топ-біржах). Причини:

- 24/7, без гепи ринку → безперервний потік даних для RL.
- Висока волатильність → швидша «крива навчання» PnL (але й ризики вищі).
- Доступний orderbook і trades на рівні tick-даних.
- Відсутність short-selling constraints (perpetual).

Архітектура має бути **мульти-інструментальною**: одна модель, що працює на N ринків одночасно, з окремими ембедінгами інструментів. Це дає transfer learning між ринками (BTC regime допомагає для ETH).

### 2.2. Таймфрейми і мультирезолюційність

Людина-трейдер дивиться на 3–6 таймфреймів одночасно (1m, 15m, 1h, 4h, 1d). Система імітує це через **multi-scale representation**:

- Кожен таймфрейм обробляється окремим (або спільним) енкодером.
- Результати агрегуються в **ієрархічний стан** (як U-Net skip-connections, але в часі).
- Down-stream задачі отримують і «мікро-контекст» (останні 100 свічок 1m), і «макро-контекст» (1–3 місяці 1d).

### 2.3. Подання часу

Час у ринку — **циклічний**, а не лінійний. Тому:

- Використовуємо **циклічні позиційні ембедінги** (sin/cos від години, дня тижня, числа місяця).
- Окремий канал — «час до наступної події» (наприклад, фідбек-вікно funding rate, вихід CPI, settlement).
- Календарний прапорник (рік/місяць) — щоб модель розуміла, що 2021 ≠ 2024.

### 2.4. Мікроструктура і orderflow

Глибина візуального сприйняття включає не тільки свічки:

- **Order book**: щільність рівнів bid/ask, дисбаланс обʼєму на рівнях, спред, мікроціни (microprice).
- **Trades flow**: агресія покупців/продавців (buy-sell imbalance за віконний час), великі угоди, кластери trades.
- **Funding rate, OI, liquidations** (для derivatives).
- **On-chain метрики** (для crypto): активність адрес, потоки на біржі, MVRV тощо.

Це все подається моделі як **числові канали** поряд із візуальними.

---

## 3. Що саме система «бачить» — візуальна таксономія

### 3.1. Цінові патерни (price action)

Система повинна розпізнавати як мінімум:

- **Trend dynamics**: імпульс, корекція, ре-тест, брейкаут, retest з обʼємом, продовження.
- **Reversal structures**: голова-плечі, подвійна вершина/дно, V-reversal, rounding bottom, island reversal.
- **Continuation structures**: прапор, вимпел, клин, трикутник (симетричний/висхідний/спадний), прямокутник.
- **Candlestick patterns**: hammer, shooting star, engulfing, tweezer, morning/evening star, marubozu, harami, three white soldiers, three black crows.
- **SMC/ICT патерни**: order blocks, fair value gaps, breaker blocks, liquidity sweeps, inducement, mitigation.
- **Wyckoff phases**: accumulation, mark-up, distribution, mark-down, spring, upthrust.

### 3.2. Обʼємні патерни

- Класичні: spike на пробої, drying up у корекціях, climax volume, volume divergence.
- Аномалії: «кишені» низького обʼєму, hidden accumulation (високий обʼєм у тілі свічки при малому діапазоні).

### 3.3. Volatility patterns

- Режим стисненої волатильності → очікуваний expansion.
- ATR squeeze, Bollinger squeeze, Keltner squeeze.
- Розширення ATR після тривалого стиснення.
- Розподіл повернень (fat tails, автокореляція).

### 3.4. Структурний контекст

- Старші таймфрейми: де ми відносно місячного/тижневого діапазону?
- Тренд старшого ТФ = зсув у бік тренду на молодшому.
- Ключові рівні: історичні high/low, VAH/VAL, daily/weekly pivot, round numbers.

### 3.5. Сентимент-індикатори (похідні)

- Funding rate extremes (over-leveraged long/short).
- Open interest change rate.
- Long/short ratio, top trader accounts.
- Social signals (якщо інтегруємо).
- Fear & Greed Index.

---

## 4. Архітектура сприйняття (perception stack)

> «Як саме raw дані перетворюються на патерни, які вже можна приймати рішення».

### 4.1. Multi-modal вхід

Система сприймає ринок у **5 каналів**:

1. **Visual chart stream** — рендер свічкового графіка у вигляді зображення (RGB), з накладеними MA, VWAP, Bollinger, обʼємом, OI. Розмірність, наприклад, 224×224×3 для базового рівня, плюс crop-и 512×512 для деталізації зон.
2. **Raw OHLCV tensor** — числовий тензор (T × F), де F — фінансові ознаки (returns, log-returns, ranges, body/wick ratios, volume z-score, distance to MA, тощо).
3. **Order book tensor** — (depth × 5) bid/ask levels з обʼємами, + зведені метрики (imbalance, microprice).
4. **Trades flow tensor** — bucketed trades за віконний час з агресією.
5. **Context vector** — час, інструмент, режим, новини (embeddings), funding, OI.

### 4.2. Енкодери

**Vision encoder** (для каналу 1):

- Backbone: ConvNeXt-V2 / EVA-02 / DINOv2 (pretrained на природних зображеннях), потім domain adaptation.
- Чому не тренувати з нуля: природні візуальні пріоритети (edges, textures, contrast) переносяться на фінансові графіки.
- Аугментації: jitter кольорів, дзеркалювання (симетрія long/short), crop-and-resize (zoom до зони інтересу), дроп рівнів, додавання шуму до обʼєму.

**Numeric encoder** (для каналів 2–4):

- PatchTST / iTransformer / N-BEATSx — сучасні архітектури для multivariate time series.
- Або спільний Transformer з різними позиційними ембедінгами для кожного тензора.
- Ціль: витягти lagged patterns, сезонність, regime indicators.

**Context encoder** (канал 5):

- MLP з categorical embeddings (інструмент, біржа) + числові (час, funding).

### 4.3. Cross-modal fusion

- Всі енкодери виходять у спільний latent простір розмірності D (наприклад, 512).
- Cross-attention Transformer: visual tokens ↔ numeric tokens ↔ context.
- Вихід fusion — **market state embedding** — компактне (D,) представлення поточного стану ринку з усіма модальностями.

### 4.4. Working memory (sequence layer)

Один стан — це мить. Для торгівлі важлива **траєкторія**.

- Над fusion-виходом працює sequence-модель: Transformer-XL / Mamba / RWKV / RetNet / Hyena — на вибір.
- Ця модель підтримує **робочу памʼять**: бачить останні K станів (наприклад, 256 барів поточного ТФ), плюс compressed memory від старших ТФ.
- Це і є «історія угоди», на яку спирається рішення.

### 4.5. Heads поверх embedding

З єдиного market state embedding «виходять» кілька голів (multi-head architecture):

- **Direction head** — supervised: напрямок на горизонті H1, H4, H24.
- **Volatility head** — supervised: майбутня реалізована волатильність (regression).
- **Regime head** — класифікація режиму (trend/range/vol_expansion/compression).
- **Risk head** — оцінка VaR / expected shortfall.
- **Policy head** — для RL: розподіл ймовірностей дій.
- **Value head** — оцінка очікуваного return зі стану.

Multi-task підхід (див. далі) тренує всі голови разом — representations стають багатшими.

---

## 5. МЕХАНІЗМ НАВЧАННЯ — ядро системи

> Це найважливіша частина. Мета — описати повний, реалістичний, стійкий до перетренування цикл навчання, який виводить агента від «порожньої сторінки» до стабільної PnL-генеруючої стратегії.

### 5.1. Філософія навчання

Жоден один лос не вирішує задачу. Використовується **5-рівнева стратифікована система навчання**, де кожен рівень додає нову здатність і спирається на попередню:

```
Рівень 1: Pre-training представлень (self-supervised)
          ↓
Рівень 2: Supervised pre-training (imitation + класифікація)
          ↓
Рівень 3: Self-play і synthetic curriculum
          ↓
Рівень 4: Reinforcement learning з реалістичним середовищем
          ↓
Рівень 5: Online continual learning + risk-aware fine-tuning
```

Нижче — кожен рівень детально.

---

### 5.2. Рівень 1 — Self-Supervised Representation Learning

**Мета:** навчити encoderи розуміти «мову ринку» без лейблів. У нас — десятки терабайтів сирих даних без анотацій. Викидати їх — нерозумно.

**Методи:**

1. **Contrastive Predictive Coding (CPC).**
   - З останніх K барів передбачаємо embedding наступного бару.
   - InfoNCE loss: позитивний приклад — справжнє майбутнє, негативні — випадкові інші бари з масиву.
   - Це вчить encoder витягати ті ознаки, які **передбачають** майбутнє.

2. **Masked modeling (як BERT для ринку).**
   - Маскуємо 30–60% каналів (random patch masking) у OHLCV + chart.
   - Енкодер відновлює замасковане. Loss — MSE для числових, cross-entropy для дискретних.
   - Це вчить encoder розуміти **взаємозвʼязки** між фічами, відновлювати контекст.

3. **Triplet / metric learning по режимах.**
   - Позитивні пари — стани з того ж режиму (тренд/range), негативні — з іншого.
   - Навчає embedding-простір бути «структурованим» за семантикою.

4. **DINO / DINOv2 self-distillation.**
   - Два енкодери (student/teacher), EMA-оновлення.
   - Chart crop → student, global chart → teacher, loss — узгодження embeddings.
   - Без лейблів виходить візуально-семантично багатий простір.

5. **Cross-modal alignment.**
   - Chart embedding і numeric embedding того ж часового відрізку мають бути близькі в просторі; різних — далекі.
   - Це змушує vision-encoder «читати» те, що вже закодовано в numbers.

**Дані:** всі доступні історичні свічки (5+ років по десятках інструментів) + chart-rendering з різних ТФ.

**Чому це працює:** encoder після цього етапу знає, де тренд, де flat, де аномалія, навіть без жодного лейблу «buy/sell». Вже цим він цінний.

---

### 5.3. Рівень 2 — Supervised Multi-Task Pre-training

**Мета:** додати знання про **наслідки** того, що відбувається на графіку. Що станеться після цього патерну? До якої категорії належить ситуація? Який ризик?

**Джерела лейблів:**

- **Автоматичні лейбли з історії (triple-barrier method):**
  - Для кожного бару t дивимось, що першим станеться протягом наступних N барів: TP_hit, SL_hit, timeout.
  - Це дає «реалістичні» outcomes, враховуючи рівні TP/SL.
  - Використовується для навчання **policy head** (класифікація triple-barrier outcomes).
- **Режимні лейбли (regime labels):**
  - Hidden Markov Model або HMM + GMM на історичних returns → latent regime (bull-trend, bear-trend, range-high-vol, range-low-vol, crisis).
  - Використовується як classification target для regime head.
- **Volatility labels:**
  - Реалізована волатильність на горизонті H (annualized std of log-returns) як regression target.
- **Risk labels:**
  - Максимальна просадка за наступні N барів, expected max adverse excursion (MAE).
- **Imitation labels (експертні трейди):**
  - Див. нижче.

**Архітектура тренування:** multi-task з shared trunk + per-task heads, зважена сума лосів:

```
L_total = Σ w_i * L_i
```

де `w_i` — зважені за:

- навчальною динамікою (gradient norm balancing — GradNorm, Uncertainty Weighting);
- відносною важливістю (Sharpe-impact of head);
- чи стабілізувався loss (EMA-based).

**Imitation learning як частина supervised:**

- Збираємо «експертні» трейди з відомих якісних джерел (публічні торгові журнали топ-трейдерів, стратегії з верифікованим track record, але також — синтетичні «експерти»).
- Для кожного моменту входу є chart + контекст + дія (long/short/skip + size).
- Тренуємо behavioral cloning head (action distribution matching).
- Проблема BC — covariate shift. Вирішується через DAgger-подібні підходи: на кожній ітерації модель тренується на станах, які вона сама відвідує.

**Навіщо imitation, якщо є RL?** Тому що:

- BC дає **стартовий поліси**, який не робить випадкових угод.
- Від нього RL стартує вже з осмисленої точки, а не з випадкової — це на порядки зменшує sample complexity.
- Imitation обходить exploration problem: модель з самого початку «знає», як виглядає реальна угода.

**Loss specifics:**

- Classification (triple barrier): focal loss (бо imbalance).
- Regression (vol): Huber / quantile (для risk-averse).
- Regime: label smoothing.
- Imitation: max-margin / cross-entropy + entropy regularization (щоб не колапсувало до детермінізму).

---

### 5.4. Рівень 3 — Self-Play Curriculum та Synthetic Worlds

**Мета:** навчити агента узагальнювати і «бачити» патерни, яких мало в реальних даних, через синтетичні середовища та curriculum.

**3.4.1. Synthetic data generation:**

Генератор ринкових симуляцій (окрема нейромережа або класична stochastic model — GARCH, jump-diffusion, Heston, regime-switching) створює реалістичні сценарії:

- Типові патерни (голова-плечі, прапор, тренд з корекціями) — у явному вигляді.
- Рідкісні події (flash crash, liquidity cascade, regime flip) — з підвищеною частотою.
- Синтетичні «торгові сесії» — з реалістичним order book evolution.

**Meta-задача:** навчити агента в synthetic environment, потім transfer на реальний ринок через:

- **Domain randomization** — у synthetic світі параметри (волатильність, спред, обʼєм) випадково варіються, щоб policy став robust.
- **Adversarial curriculum** — generator світу адаптується: створює саме ті сценарії, де агент помиляється.

**3.4.2. Self-play і counterfactual reasoning:**

- Модель «уявляє» альтернативні траєкторії: «якби я не увійшов тут, а увійшов 3 бари пізніше?».
- Реалізується через learned world model (див. 5.5) або Monte Carlo rollouts.
- Це вчить агента розрізняти хороші і погані входи в близьких ситуаціях — критично для точності.

**3.4.3. Curriculum learning:**

Тренування йде від простого до складного:

- Етап 1: чіткі тренди з мінімумом шуму (синтетика).
- Етап 2: тренди + корекції.
- Етап 3: тренди + ranges + volatility events.
- Етап 4: повні реалістичні дані.
- Етап 5: stressed periods (бектест на 2020 COVID, 2022 Luna, 2024 ETF flows).

Кожен наступний етап стартує з policy попереднього — curriculum transfer.

---

### 5.5. Рівень 4 — Reinforcement Learning у реалістичному середовище

> Це серце системи. Тут policy «заточується» під реальні PnL.

**5.5.1. Середовище (Environment):**

Не сирий ринок, а **симулятор виконання**, який точно моделює:

- Latency (мережева затримка до біржі, наприклад 50–300 мс).
- Orderbook dynamics: прийняття/скасування ордерів, queue position.
- Slippage model: залежно від обʼєму ордера і глибини книги.
- Spread dynamics.
- Funding payments (perpetual).
- Liquidation mechanics.
- Fees (maker/taker).

**Типи ордерів у action space:**

- Market, limit, post-only, IOC.
- Reduce-only.
- Stop-loss, take-profit (як окремі обʼєкти стану, не тільки закриття).

**Action space:**

Дискретний для простоти: {skip, long_25%, long_50%, long_100%, short_25%, short_50%, short_100%, close, partial_close}.

Або неперервний: (direction ∈ [-1,1]) × (size ∈ [0,1]) × (sl_atr_mult ∈ [0.5,5]) × (tp_atr_mult ∈ [0.5,10]).

**5.5.2. Reward shaping — критичний дизайн:**

Чистий PnL — погана винагорода (висока дисперсія, зловживання «all-in bets»). Тому reward — **сконструйований** ризик-скоригований сигнал:

```
r_t = ΔEquity_t
      - λ_drawdown * max(0, peak_eq - eq_t)
      - λ_variance * Var(returns, last W)
      - λ_turnover * |Δposition|
      + λ_sharpe_bonus * rolling_sharpe_increment
      - λ_liquidation * liquidations_penalty
      - λ_slippage * realized_slippage_cost
```

Додаткові компоненти:

- **Survival bonus** (маленький +) за те, що equity не падає нижче threshold — щоб policy не «здавалась» після drawdown.
- **Risk-adjusted return** (Sharpe / Sortino за rolling window) як основний сигнал.
- **Penalties for over-trading** — щоб policy не «смикалась» на кожному барі.

**Цільова функція:**

Не просто max Σ r_t, а max **CVaR-aware utility** (conditional value at risk). Наприклад:

- Мета: maximize E[U(PnL)] де U — увігнута функція (CARA / CRRA).
- Constraint: P(drawdown > X) < Y.
- Реалізується через Lagrangian-методи в RL.

**5.5.3. Алгоритми RL:**

Дві гілки паралельно:

1. **PPO / SAC** — для stability та ease of deployment. PPO з GAE, кліпована цільова.
2. **Decision Transformer / Trajectory Transformer** — modeling policy як sequence modeling над (state, return-to-go, action). Дуже добре для offline RL (див. 5.5.5).
3. ** distributional RL** (IQN, FQF) — для кращого model uncertainty.
4. **Model-based RL (DreamerV3-style)** — окремий світ-модель, rollouts у мріях, реальні дані для calibration.

**5.5.4. Replay buffer та off-policy:**

- Досвід зберігається у **prioritized replay** з пріоритетами за TD-error і за «цікавістю» (states, де policy невпевнена).
- Окремий **expert replay buffer** — найкращі епізоди з минулого, які policy постійно переграє.
- **Hindsight Experience Replay (HER)** — адаптований: навіть якщо угода закрилась у мінус, ретроспективно «переграємо» її як приклад для задачі «як закритись у плюс у такій ситуації» (multi-goal RL).

**5.5.5. Offline-to-online RL:**

- Спочатку — **offline RL** на величезному масиві історичних trades (наш власний historical order book + ідентифікація того, «що б ми зробили»).
- Потім — **online fine-tuning** у paper trading.
- Online-фаза обережна: low learning rate, frequent evaluation gates, kill-switch при просіданні.

**5.5.6. Risk-aware обмеження:**

- **CVaR constraint**: PnL на 5-му перцентилі має бути > -X% від equity.
- **Max leverage** (наприклад, 3x) — hard constraint у action space.
- **Max position per instrument** — diversification constraint.
- **Daily loss limit** — mandatory stop, не рекомендовано.

---

### 5.6. Рівень 5 — Online Continual Learning і Self-Improvement

**Проблема:** ринок змінюється. Модель, натренована вчора, завтра може бути субоптимальною. Ми не можемо зупинити навчання.

**Підходи:**

1. **Sliding-window fine-tuning.** Кожну добу/тиждень — короткий fine-tune на останніх N даних, з regularization до старої policy (EWC, KL-divergence constraint). Це запобігає catastrophic forgetting.
2. **Streaming evaluation gates.** Перш ніж прийняти нову версію моделі, вона проходить paper-trading у паралельному режимі 24–72 години. Якщо Sharpe на paper нижчий за baseline — rollback.
3. **Concept drift detection.** Окрема модель моніторить розподіл returns і «дивні» стани. Сповіщає, коли ринок увійшов у новий regime.
4. **Meta-learning.** Навчити policy бути «адаптивним за кілька кроків»: MAML / Reptile поверх RL policy, щоб нова під-адаптація до нового режиму вимагала лише 100–1000 прикладів.
5. **Population-based training (PBT).** Утримуємо популяцію з N policy-варіантів, кожен з гіперпараметрами; «виживають» найкращі, мутують — еволюційний тиск.

**5.6.1. Online reward calibration:**

Reward function **не статична**. Її ваги λ_* — теж learnable, але обережно:

- Динамічно зсуваються в бік консервативності при підвищенні VIX / realized vol.
- Переоцінюються щотижня на основі останніх результатів policy.

**5.6.2. Self-distillation та self-improvement:**

- Найкраща за останній період policy стає «teacher» для нових policy-ей.
- Дистиляція поверх різних ініціалізацій — пошук кращих локальних оптимумів.

---

### 5.7. Додаткові критичні компоненти навчання

**5.7.1. Uncertainty estimation.**

Модель має знати, **коли вона не знає**:

- **MC-Dropout** або **Deep Ensembles** для epistemic uncertainty.
- **Evidential regression** для aleatoric.
- У action selection: пропускати угоди при високій невизначеності (risk-off).
- Це критично: одна з головних причин збиткових ботів — відсутність «I don’t know»-сигналу.

**5.7.2. Calibration.**

- Predicted probabilities мають відповідати реальним частотам.
- Expected Calibration Error (ECE) < 5% — обовʼязкова метрика.
- Використовується temperature scaling, isotonic regression.

**5.7.3. Robustness і adversarial training.**

- Додаємо контрольовані збурення в inputs (adversarial examples) — вчить модель не ламатись від шуму даних.
- AugMix-style аугментації для chart-rendering.

**5.7.4. Interpretability / Mechanistic analysis.**

- Attention rollout — куди дивиться vision encoder.
- Saliency maps на inputs.
- SHAP для numeric features.
- **Це не косметика, а debugging tool**: якщо модель «торгує» зовсім не з тих причин, які мали б сенс — це black flag.

**5.7.5. Anti-overfit techniques (повний список):**

- Walk-forward validation (rolling origin), **не random split**.
- Purged cross-validation з embargo (див. López de Prado).
- Dropout, weight decay, stochastic depth.
- Heavy data augmentation.
- Curriculum і progressive resizing (як у CV).
- Label smoothing.
- Early stopping за out-of-sample Sharpe, не за training loss.
- Snapshot ensembling.
- Stochastic weight averaging (SWA).

**5.7.6. Anti-look-ahead bias.**

- Усі features — strict lagged.
- Triple barrier labeling — з real-time правилами (TP/SL оцінюються тільки по майбутніх close, не знаючи проміжних).
- Train/test split — часовий, з додатковим gap (embargo) щоб уникнути leak через autoreg.

**5.7.7. Counterfactual evaluation.**

- Backtest завжди включає counterfactual: «а якби ми не торгували в день X?». Це валідує, що PnL — від policy, а не від ринку.

---

## 6. Пайплайн даних

### 6.1. Збір

- **Historical OHLCV**: ≥5 років для топ-інструментів, m1+ resolution.
- **Historical order book snapshots**: збережені L2 snapshots якомога далі (або відновлені з trades через queue models).
- **Historical trades (tick)**: aggTrades для крипти, ticks для tradfi.
- **Funding, OI, liquidations**: щоденні/щогодинні часові ряди.
- **On-chain**: stablecoin supply, exchange netflow, BTC/ETH specific metrics.
- **Macro/News**: embeddings з новинних заголовків, calendar of events.

### 6.2. Зберігання

- **Time-series DB**: TimescaleDB / InfluxDB / QuestDB.
- **Feature store**: окремий layer для pre-computed features.
- **Image rendering cache**: згенеровані chart images (через mplfinance / lightweight-charts) зберігаються у Parquet / WebP.

### 6.3. Augmentation

- **Time warping** (resample до 1.1×, 0.9×).
- **Jitter** обʼєму.
- **Mirror** (long/short swap).
- **Color jitter** для charts.
- **Mixup** на сусідніх станах.
- **CutMix** для chart regions.

### 6.4. Quality control

- Аномалії: gaps, NaNs, нуль-обʼємні бари — фільтруються.
- Survivorship bias: враховувати delisted інструменти (для tradfi).
- Look-ahead audit: автоматичні тести, що жоден feature не «бачить» у майбутнє.

---

## 7. Архітектура прийняття рішень

### 7.1. Повний pipeline (inference time)

```
[raw streams]
   ↓
[sync to 1s clock, normalize, align]
   ↓
[render chart images @ multiple TFs]
   ↓
[encode all modalities → market state embedding]
   ↓
[working memory: temporal context]
   ↓
[heads: direction, vol, regime, risk]
   ↓
[policy head: action distribution]
   ↓
[risk gates: max position, exposure, daily loss, uncertainty]
   ↓
[order construction: entry, SL, TP, size]
   ↓
[execution: smart order routing]
   ↓
[monitoring, PnL attribution]
```

### 7.2. Decision frequency

Не кожен бар — рішення. Decision head активується лише на певних барах (наприклад, на закритих свічках вибраного ТФ, або на нових станах order book). Це запобігає over-trading.

### 7.3. Multi-instrument portfolio

- Один глобальний policy, але з **інструмент-ембедінгом** як частиною state.
- У agent — пул інструментів. На кожному кроці вибирає інструмент + дію (або додатковий дискретний action «switch instrument»).
- Або: окрема голова пропонує allocation per instrument, потім — single-instrument trading.

### 7.4. Order management

- **Піраміда**: scale-in, scale-out.
- **Trailing stop**: ATR-based або chandelier.
- **Time stop**: якщо угода не рухається N барів — закрити.
- **Re-entry policy**: якщо угода закрилась по SL і setup все ще валідний — переоцінити.

---

## 8. Метрики, оцінка, бектест

### 8.1. Метрики performance

- **Sharpe ratio** (річний, annualized).
- **Sortino ratio**.
- **Calmar ratio** (return / max DD).
- **Win rate**, **profit factor**, **expectancy per trade**.
- **Tail metrics**: max DD, max DD duration, recovery factor.
- **Stability**: Sharpe у часі (rolling 30/60/90 днів), плюс fraction часу з позитивним rolling Sharpe.
- **Deflated Sharpe Ratio** — коригуємо на кількість спроб / strategies.

### 8.2. Benchmarks

- Buy & hold BTC perp.
- 60/40 (BTC/ETH).
- SMA crossover.
- Equal-weight топ-3 momentum.

### 8.3. Процедура оцінки

1. **In-sample**: train на 2019–2022.
2. **Out-of-sample**: 2023 (ембарго 1 тиждень).
3. **Walk-forward**: rolling origin retraining щотижня.
4. **Monte Carlo**: 1000+ ресемплів historical PnL для розподілу метрик.
5. **Stress tests**: COVID 2020, Luna 2022, FTX 2022, ETF approval Jan 2024, корреляція з TradFi sell-offs.
6. **Paper trading**: 1+ місяць out-of-sample, з реальним latency та execution.
7. **Micro-live**: мінімальний size (0.1% equity), зростання розміру по мірі довіри.

### 8.4. Статистична значущість

- Bootstrap CIs на Sharpe.
- Permutation tests для перевірки, що результат не випадковий.
- Bayesian Sharpe: posterior over true Sharpe, не лише point estimate.

---

## 9. Управління ризиками (production-grade)

### 9.1. Hard constraints (invariants)

- **Max leverage**: 3x.
- **Max gross exposure**: 100% equity (no over-leverage).
- **Max per-trade risk**: 1% equity (1R).
- **Daily loss limit**: -3% → stop trading 24h.
- **Weekly loss limit**: -5% → режим зниженого розміру.
- **Drawdown kill-switch**: -15% from peak → перейти в 100% cash.

### 9.2. Soft constraints (learned)

- CVaR constraint.
- Correlation-aware sizing (не over-concentrate в одному напрямку).
- Volatility targeting: σ_target = 20% річних, size інверсно до realized vol.

### 9.3. Execution risk

- Order types: prefer post-only, limit.
- TWAP / VWAP для великих ордерів.
- Iceberg orders для уникнення сигналування.
- Failover біржі.

### 9.4. Operational risk

- Multi-region deployment.
- Circuit breakers на API-помилки.
- Manual kill-switch.
- Alerting на аномалії (P&L spike, fills rate drop).

---

## 10. Технічна інфраструктура (концептуально)

### 10.1. Data layer

- TimescaleDB (PostgreSQL extension) для time-series.
- S3-сумісне сховище для chart images, parquet snapshots.
- Feature store (Feast або кастомний).

### 10.2. Training infra

- Multi-GPU training (A100/H100) для vision + sequence моделей.
- Ray / Dask / SLURM для distributed training.
- MLflow / Weights & Biases для experiment tracking.
- Model registry, lineage.

### 10.3. Inference infra

- Real-time сервіс: low-latency inference (ONNX / TensorRT).
- Streaming input (Kafka / Redpanda / NATS JetStream).
- Model serving з hot-reload.

### 10.4. MLOps

- CI/CD для моделей (data validation, model validation, canary deploy).
- Feature drift, prediction drift monitoring.
- Auto-rollback за метриками.

---

## 11. Дорожня карта (фази)

### Фаза 0 — Foundations (1–2 місяці)
- Збір і нормалізація даних.
- Побудова chart-rendering pipeline.
- Бейзлайн-стратегії для ground truth (SMA cross, momentum).
- Backtester (як окремий модуль з реалістичним execution).

### Фаза 1 — Self-Supervised Pre-training (2–3 місяці)
- CPC, masked modeling, DINO на charts.
- Self-supervised numeric encoder.
- Cross-modal alignment.
- Оцінка якості embeddingів через downstream probing.

### Фаза 2 — Supervised Multi-Task (2–3 місяці)
- Triple barrier labeling.
- Multi-task training (direction, vol, regime, risk).
- Imitation learning з public signals.
- Досягнення baseline якості на walk-forward.

### Фаза 3 — Synthetic Curriculum (1–2 місяці)
- Market simulator.
- Adversarial data generation.
- Curriculum training.
- Domain randomization.

### Фаза 4 — RL Policy (3–4 місяці)
- Realistic execution simulator.
- PPO/SAC базовий RL.
- Reward shaping experiments.
- Risk-aware variants.

### Фаза 5 — Online / Continual (ongoing)
- Paper trading з fine-tuning.
- A/B різних policies.
- Concept drift handling.
- Population-based training.

### Фаза 6 — Production (post-validation)
- Micro-live (0.1% size).
- Scale-up за результатами.
- Multi-account, multi-region.

---

## 12. Гіпотези, що будуть перевірятись

1. Візуальний енкодер кращий за чисто числовий для виявлення reversal патернів.
2. Multi-task покращує representations для trading policy.
3. Imitation bootstrap суттєво зменшує sample complexity RL.
4. Synthetic curriculum покращує generalization на stressed periods.
5. Risk-adjusted reward → менш volatile equity curve ніж raw PnL reward.
6. Decision Transformer перевершує PPO на offline historical trades.
7. Uncertainty-aware abstention зменшує DD.
8. Cross-instrument transfer (BTC → ETH) реально працює.

---

## 13. Ризики проекту (мета-ризики)

1. **Ефективність ринку (EMH)**: можливо, edge не існує на 1m/5m через HFT. На 1h/4h/daily — більше шансів.
2. **Регуляторні ризики**: залежно від юрисдикції.
3. **Технічні ризики**: data corruption, API outage, latency spikes.
4. **Ментальні пастки**: переоптимізація, gamblerʼs fallacy, невірне тлумачення backtest equity.
5. **Survivorship bias в training data**: враховувати delisted інструменти / dead tokens.
6. **Мікроструктурна зміна**: біржі змінюють fee schedule, matching engine — модель має перенавчатись.

---

## 14. Фінальні принципи

- **Жоден лос не вирішує задачу** — тільки комбінація self-supervised + supervised + RL + continual.
- **Складність виправдана, якщо вона додає robustness, а не overfit**. Тому жорстка регуляризація і walk-forward evaluation.
- **Risk-aware все**: від reward function до kill-switch.
- **Interpretability — не опція**. Чорний ящик у production = нерозумний ризик.
- **Прибуток — функція процесу, а не функція моделі**. Дисципліновані бектести → paper → micro → scale.

---

## 15. TL;DR (одна сторінка)

**ZHISA** — це мультимодальний, ієрархічний, ризик-скоригований торговий агент, що:

- **Бачить** графіки як людина-трейдер (vision encoder на rendered charts).
- **Розуміє** контекст через numeric encoders (OHLCV, order book, trades flow, on-chain).
- **Памʼятає** через sequence-модель робочої памʼяті.
- **Навчається** у 5 фаз: self-supervised → supervised multi-task → synthetic curriculum → RL → online continual.
- **Вирішує** задачу як RL з реалістичним execution-симулятором і risk-shaped reward.
- **Існує** у production-grade інфраструктурі з моніторингом, kill-switchʼами, continual retraining.
- **Вимірює себе** за risk-adjusted метриками, на walk-forward + out-of-sample + stress + paper, а не за training loss.

Глибина і складність — функція вимоги до robustness. Мета не «зловити рух», а побудувати стійкий, інтелектуальний, еволюціонуючий агент, що реально торгує в нестаціонарному середовищі — і при цьому залишається пояснюваним і керованим.
