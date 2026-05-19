# Hipótesis high reward / low frequency para trading sistemático

## Supuestos operativos

Interpreto **+50%/+100%** como retorno sobre **capital arriesgado**, no sobre NAV total. Para opciones: prima/debit neto. Para acciones/ETF: distancia a stop o pérdida máxima modelada, aunque con riesgo de gap. En backtest, todas las entradas deben ser **siguiente open**, **siguiente close**, o con timestamp intradía real; no usar fills al mismo close si la señal se calcula con ese close.

Datos razonables: Massive/Polygon permite trabajar con opciones, quotes/trades históricos, agregados, IV/greeks/open interest según plan; Cboe DataShop ofrece trades de opciones con NBBO, bid/ask del subyacente y greeks opcionales; FINRA publica short-sale volume agregado por TRF/ADF/ORF el mismo día; OCC publica reports de volumen/open interest; SEC publica fails-to-deliver, pero advierte que son balances agregados, no “fails nuevos del día”, y que no permiten inferir edad del fail.

Reglas comunes de ejecución para las 15 hipótesis: no operar opciones con spread efectivo superior al 8–12% de la prima salvo eventos binarios; exigir OI mínimo 100–500 contratos y volumen diario mínimo 25–250 según liquidez; simular fills a **ask para entrada long**, **bid para salida**, o mid ± 0.5 spread; limitar tamaño a <2–5% del volumen diario de la opción y <1% del ADV del subyacente; reportar resultados con y sin el mejor 1% de trades.

Controles comunes antifraude estadístico: predefinir umbrales antes del test final, walk-forward temporal, embargo de eventos solapados, universo con delistings, corporate actions ajustadas, costes conservadores, bootstrap por fecha y ticker, comparación contra baselines, y separar claramente **screening**, **validación** y **test cerrado**.

---

# 1. Alta prioridad: investigables pronto con datos razonables

## H1 — Vol ETP Panic Decay Put

**Instrumento recomendado:** puts o put debit spreads sobre UVXY/VXX; alternativa: SVIX/SVXY solo si se acepta riesgo de gap inverso. Mejor estructura inicial: put vertical 45–90 DTE, long ATM o 5% OTM, short 25–40% OTM.

**Tipo de payoff:** long convexity sobre colapso de volatilidad; estructural decay; volatility mean reversion.

**Mecanismo económico:** tras un shock de volatilidad, los ETP long vol quedan expuestos a reversión de VIX futures, roll decay y compresión de IV. El payoff no depende de acertar muchos trades pequeños: pocos episodios donde UVXY/VXX caen 30–70% pueden pagar muchas primas perdidas. Cboe documenta que VIX se basa en opciones SPX/SPXW y Cboe lista futuros VIX, por lo que el estado de la curva VX es observable.

**Evento/señal:** VIX > percentil 90 de 1 año o VIX > 30; VX1/VX2 en backwardation; después, primer día en que la backwardation se reduce y SPX deja de marcar nuevo mínimo de 3 días. Señal conservadora: VIX cae ≥10% desde máximo reciente y UVXY no consigue nuevo máximo.

**Entrada candidata:** siguiente open o close tras confirmación. Comprar put vertical 45–90 DTE. Filtro: spread de la opción <8% de prima, OI >500, volumen >250 contratos.

**Salida candidata:** +100% sobre debit, 50% del máximo beneficio del vertical, retorno de VX1/VX2 a contango, o 20–30 sesiones. Stop opcional: vender si la prima cae 50–70%; pérdida máxima real: debit.

**Qué podría generar +50%/+100%:** caída rápida de UVXY/VXX por compresión de VIX, normalización de curva VX y theta favorable en el put.

**Frecuencia esperada:** 1–4 clusters/año, no necesariamente todos operables.

**Datos necesarios:** OHLCV UVXY/VXX, opciones EOD o intradía, VIX, VIX futures curve, quotes.

**Dificultad de backtest:** media.

**Riesgos de sesgo/look-ahead:** usar VX settlement posterior; seleccionar expiries inexistentes; ignorar reverse splits en ETPs; fills demasiado optimistas en opciones.

**Baselines obligatorios:** short UVXY con stop; put vertical después de cualquier VIX >30; compra aleatoria de puts tras shocks; short VXX/UVXY delta-equivalente.

**Controles antifraude:** test por episodios de crisis, no por trades individuales; excluir el mejor trade y recalcular; sensibilidad a 1, 2 y 3 días de retraso en entrada.

**Por qué podría fallar:** otro crash encadena backwardation, UVXY sube violentamente, IV de puts ya descuenta el decay, o el spread/options slippage elimina convexidad.

---

## H2 — VIX Complacency Break Call Spread

**Instrumento recomendado:** VIX call spreads o call broken-wing spreads 30–60 DTE. Ejemplo: VIX 15/25, 17.5/30 o 20/35 según spot. Evitar calls naked demasiado OTM si el spread es ancho.

**Tipo de payoff:** long convexity; volatility expansion; crash lottery controlada.

**Mecanismo económico:** cuando la volatilidad implícita está comprimida y hay divergencias de estrés, el mercado de short-vol puede estar vendiendo seguro barato. Un salto de VIX produce repricing no lineal. VIX mide volatilidad esperada de 30 días vía opciones SPX/SPXW, por lo que la señal puede anclarse en datos de mercado observables.

**Evento/señal:** VIX bajo o medio-bajo, por ejemplo <16 o percentil <35; SPX cerca de máximos; pero HYG/LQD débil frente a SPY, USDJPY/FXY o TLT rompiendo estrés, o VVIX/VIX subiendo. Requiere divergencia, no simple “VIX bajo”.

**Entrada candidata:** comprar call spread 30–60 DTE cuando VIX cierra por encima de su media de 5 días tras estar comprimido y SPX pierde mínimo de 10 días. Filtro de liquidez: bid/ask del spread <12% del debit.

**Salida candidata:** +100% sobre debit, VIX +5/+8 puntos, SPX recupera máximo previo, o 10 días antes de expiry. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** salto de VIX de 15 a 25–35, ampliación de VVIX y repricing de convexidad.

**Frecuencia esperada:** 2–6 señales/año; muchas serán falsas.

**Datos necesarios:** VIX, VIX options, SPX/SPY OHLCV, HYG/LQD, TLT, FXY/USDJPY, opcional VVIX.

**Dificultad de backtest:** media-alta por opciones VIX.

**Riesgos de sesgo/look-ahead:** VIX options liquidan contra futuros, no spot VIX; usar spot como si fuera entregable distorsiona; vencimientos y strikes deben existir en la cadena.

**Baselines obligatorios:** call spreads VIX comprados mensualmente; SPY puts con mismo debit; señal solo con VIX bajo; señal solo con SPX breakdown.

**Controles antifraude:** out-of-sample por régimen de volatilidad; sensibilidad a strike; probar 30, 45 y 60 DTE sin seleccionar ex post; capar payoff extremo.

**Por qué podría fallar:** la divergencia no importa, el mercado vende vol correctamente, el VIX sube tarde, o la estructura pierde por contango/time decay.

---

## H3 — Reg SHO Forced-Cover Squeeze

**Instrumento recomendado:** acciones para nombres ilíquidos; calls 30–60 DTE solo si las opciones son razonables. Mejor contrato: call 0.35–0.55 delta o call vertical 10–30% OTM.

**Tipo de payoff:** squeeze; forced buy-in; gap follow-through.

**Mecanismo económico:** una security en threshold list implica fails-to-deliver persistentes. Regulation SHO contempla close-out y pre-borrow requirements; Nasdaq define threshold security por fails de al menos 10,000 shares y ≥0.5% de shares outstanding durante cinco settlement days. Esto crea una posible presión de compra o restricción a nuevos shorts, aunque no todos los casos son squeezes.

**Evento/señal:** ticker entra en Reg SHO threshold list y permanece ≥5–10 días; precio no colapsa; volumen relativo >3x; close en top 40% del rango diario; float/ADV bajo. Evitar OTC, precio <$2, reverse split reciente y tickers con offering activo.

**Entrada candidata:** siguiente open tras lista publicada y confirmación de precio. Para acciones: stop bajo mínimo del día de señal o -10/-15%. Para opciones: calls 30–60 DTE con spread <12%, OI >100, volumen >50.

**Salida candidata:** +2R/+5R en acciones, trailing stop bajo mínimo de 2 días, salida tras salida de threshold list, o 10–15 sesiones. En opciones: +100%/+200% o pérdida de 50% de prima.

**Qué podría generar +50%/+100%:** short covering, buy-ins, incapacidad de abrir nuevos shorts sin borrow, gamma si hay calls activos.

**Frecuencia esperada:** varias señales/mes en universo amplio; alta calidad quizá 1–4/mes.

**Datos necesarios:** threshold lists, OHLCV, opciones, float, short interest con lag, SEC FTD solo como feature rezagada. FINRA short-sale volume puede ayudar, pero no equivale a short interest.

**Dificultad de backtest:** media.

**Riesgos de sesgo/look-ahead:** usar FTD publicados con retraso como si fueran conocidos; survivorship en microcaps; no respetar publicación post-close de la lista.

**Baselines obligatorios:** comprar todos los Reg SHO sin filtro; comprar high relative volume no-Reg SHO; comprar al primer día de lista frente a día 5/10; acciones frente a calls.

**Controles antifraude:** excluir tickers con market cap muy baja; repetir con filtros de liquidez crecientes; cluster bootstrap por fecha; medir resultado sin top 5 squeezes.

**Por qué podría fallar:** threshold por problemas operativos, dilución, fraude, manipulación, opciones carísimas o imposibilidad real de ejecutar tamaño.

---

## H4 — Multi-Halt Momentum Continuation

**Instrumento recomendado:** acciones; opciones solo en large/mid caps con spreads estrechos. Evitar microcaps con spreads extremos salvo investigación separada.

**Tipo de payoff:** squeeze intradía; order-book imbalance; gap follow-through.

**Mecanismo económico:** múltiples LULD halts o halts por news muestran desequilibrio extremo de órdenes. Shorts y market makers pueden quedar atrapados, y la reapertura crea feedback de liquidez. NYSE y Nasdaq publican datos de trading halts; NYSE indica histórico de news/LULD de un año, y Nasdaq tiene páginas de halt history.

**Evento/señal:** acción $2–$50, ADV dollar >$10M, subida intradía >20%, ≥2 LULD halts alcistas o halt T1/T2 con reapertura por encima del VWAP intradía. Excluir reverse splits recientes, warrants, SPACs problemáticos y float ínfimo.

**Entrada candidata:** tras reapertura, si primera vela de 5 minutos mantiene precio por encima del nivel de halt/reopen y volumen sigue >5x. Entrada con bracket: stop 8–15% o bajo VWAP/reopen.

**Salida candidata:** parcial en +2R, trailing bajo mínimos de 5 minutos, cierre EOD, o salida en premarket siguiente si gap-up. Pérdida máxima: stop, pero con riesgo de halt/gap.

**Qué podría generar +50%/+100%:** movimientos intradía de 50–200% con stop inicial de 8–15%; calls pueden multiplicarse si hay liquidez.

**Frecuencia esperada:** muchas señales brutas; con filtros serios, 2–5/mes.

**Datos necesarios:** intraday OHLCV, trades/quotes, halt feed, corporate actions, opciones si se usan.

**Dificultad de backtest:** media-alta por timestamps y halts.

**Riesgos de sesgo/look-ahead:** contar halts posteriores para decidir entrada; usar high del día; ignorar reopen spreads; usar histórico incompleto.

**Baselines obligatorios:** comprar top gainers intradía sin halts; comprar primer halt versus segundo halt; esperar 5/15/30 minutos; no operar opciones.

**Controles antifraude:** slippage severo, fills parciales, simulación con delay de 1 minuto, eliminar microcaps, reportar capacidad máxima.

**Por qué podría fallar:** el halt marca distribución, la reapertura abre con gap contra la posición, spreads imposibles, o la señal se degrada por competencia.

---

## H5 — Capitulation Gap Reversal

**Instrumento recomendado:** acciones con stop definido; call debit spread 21–45 DTE si las opciones siguen líquidas. En small/mid caps, shares suelen ser más backtesteables.

**Tipo de payoff:** reversion extrema; short-covering; forced liquidation rebound.

**Mecanismo económico:** una caída de 30–50% en una acción líquida puede venir de ventas forzadas, margin calls, liquidación de fondos o sobrerreacción. Si el precio rechaza los mínimos, el rebote puede ser violento frente a un stop relativamente cercano.

**Evento/señal:** gap-down o caída diaria >30%, o caída de 3 días >45%; volumen >10x ADV; close en top 40% del rango intradía; precio >$5; ADV dollar >$20M; no OTC; no bankruptcy proxy obvio.

**Entrada candidata:** siguiente open si no abre por debajo del mínimo de capitulación. Para shares: stop bajo mínimo del evento. Para opciones: call vertical ATM/10–20% OTM, 21–45 DTE, spread <10%.

**Salida candidata:** +2R/+4R, recuperación de 50% del gap, VWAP de evento, media de 10/20 días, o 3–10 sesiones. Pérdida máxima: stop o debit.

**Qué podría generar +50%/+100%:** rebote de 20–70% con stop de 5–12%; en calls, expansión delta/gamma aunque IV se comprima.

**Frecuencia esperada:** 1–3 señales/mes en universo US líquido; alta calidad menos.

**Datos necesarios:** OHLCV ajustado, intraday high/low, delistings, corporate actions, opciones opcionales, filings/news para exclusiones.

**Dificultad de backtest:** baja-media.

**Riesgos de sesgo/look-ahead:** survivorship; excluir fraudes usando noticias conocidas después; fills al close del mismo día; gaps bajo stop.

**Baselines obligatorios:** comprar todos los gap-downs; comprar gap-down sin close-in-range; esperar 1 día adicional; short continuation como control.

**Controles antifraude:** incluir delisted; probar con entrada next open y next close; capar retornos; separar earnings, FDA, fraud y macro.

**Por qué podría fallar:** falling knife real, fraude, dilución, covenant breach, IV demasiado alta o gap adicional contra stop.

---

## H6 — Call-Wall Gamma Impulse

**Instrumento recomendado:** acciones líquidas o call verticals 7–21 DTE. Contrato aproximado: long call 0.40–0.60 delta, short call 10–25% superior para financiar IV.

**Tipo de payoff:** gamma squeeze; call-flow continuation; reflexive hedging.

**Mecanismo económico:** cuando hay call volume/OI extremo en strikes cercanos y el precio cruza esos strikes, dealers potencialmente short gamma pueden comprar subyacente para cubrir. El mecanismo no es “breakout técnico”; el breakout es proxy de hedging y positioning.

**Evento/señal:** call volume/OI del día >1.5–3 en opciones 0–14 DTE; volumen total de calls >5x mediana 20 días; subyacente cierra por encima del strike de mayor OI/call volume cercano; relative volume >3x; precio >$10 y ADV dollar >$50M.

**Entrada candidata:** siguiente close/open tras cruce confirmado. Preferir shares si el spread de opciones explota. En opciones: 7–21 DTE, call vertical, spread neto <10%, OI >500.

**Salida candidata:** +100% en prima, expiry -2 días, pérdida del strike clave, o 3–7 sesiones. Pérdida máxima: debit; en shares, stop bajo strike/vwap.

**Qué podría generar +50%/+100%:** aceleración por hedging, FOMO, short covering y gamma cerca de expiry.

**Frecuencia esperada:** 2–6/mes tras filtros; menos en large caps estrictos.

**Datos necesarios:** opciones volume/OI por strike, quotes, OHLCV, opcional short interest/borrow.

**Dificultad de backtest:** media.

**Riesgos de sesgo/look-ahead:** OI se actualiza T+1; no inferir OI intradía; identificar spreads complejos como directional flow puede ser falso.

**Baselines obligatorios:** price breakout sin call-flow; call-flow sin cruce de strike; random high call volume; shares frente a calls.

**Controles antifraude:** usar OI previo al día de señal; probar delay de 1 día; filtrar block trades/spreads si se dispone; no optimizar strike exacto.

**Por qué podría fallar:** el flow era closing/covered calls, dealers estaban long gamma, IV crush, o el movimiento ya está agotado.

---

## H7 — Duration Shock Convexity

**Instrumento recomendado:** opciones TLT 45–90 DTE; alternativa más limpia: opciones sobre Treasury futures ZN/ZB si tienes data CME. Para empezar: TLT put/call vertical ATM ±5–10%.

**Tipo de payoff:** rates trend expansion; volatility expansion; macro convexity.

**Mecanismo económico:** rupturas de rango en yields pueden forzar duration hedging, convexity hedging hipotecario y repricing de política monetaria. TLT options permiten pérdida limitada sin entrar directamente en futures.

**Evento/señal:** TLT realized vol 40d en percentil <25; rango 20d comprimido; cierre fuera de rango 90d; confirmación por 10Y yield rompiendo máximo/mínimo 90d. Dirección: TLT puts si yields rompen arriba; calls si yields rompen abajo.

**Entrada candidata:** siguiente open/close; comprar 45–90 DTE ATM option o debit spread 5–10% ancho. Filtro: spread <6–8%, OI >500.

**Salida candidata:** +100% sobre debit, retorno al rango, 20 sesiones, o expiry -15 días. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** movimiento de TLT de 5–12% con subida de IV de rates.

**Frecuencia esperada:** 2–5/año.

**Datos necesarios:** TLT OHLCV/options, Treasury yields, opcional MOVE, CME futures/options volume/open interest. CME publica reports de volumen y open interest, con data oficial diaria al día siguiente.

**Dificultad de backtest:** baja-media.

**Riesgos de sesgo/look-ahead:** timestamps de yields frente a ETF close; vencimientos de opciones; anuncios FOMC conocidos ex ante pero no resultado.

**Baselines obligatorios:** TLT shares con stop; straddles en FOMC; breakout TLT sin compresión; random 45D options.

**Controles antifraude:** split por regímenes de tipos; test con yields y solo con TLT para ver dependencia; sensibilidad 30/60/90 DTE.

**Por qué podría fallar:** ruptura falsa, Fed/treasury supply ya descontado, TLT options caras, o reversión por risk-off.

---

## H8 — Yen Carry Unwind Proxy

**Instrumento recomendado:** FXY calls 60–120 DTE si liquidez suficiente; alternativa: JPY futures options o USDJPY options vía broker si puedes capturar quotes. Shares FXY con stop si opciones son malas.

**Tipo de payoff:** carry unwind; FX volatility expansion; macro squeeze.

**Mecanismo económico:** el yen suele actuar como funding currency. Cuando se rompe el carry por caída de risk assets o compresión de diferenciales de tipos, el unwind puede ser brusco. La convexidad viene de comprar opciones antes de que el mercado reprima carry risk.

**Evento/señal:** USDJPY extendido en máximo 120d; SPX cae >3% en 5 días o VIX sube; diferencial US-Japan 2Y deja de ampliar; FXY cierra por encima de máximo 20d tras tendencia bajista previa.

**Entrada candidata:** FXY call 60–120 DTE, 0.35–0.55 delta, o call vertical 5–10% ancho. Filtro estricto: spread <10–12%, OI >100; si no, operar FXY shares con stop bajo mínimo 10d.

**Salida candidata:** +100% en opción, USDJPY revierte 50% del impulso, FXY pierde mínimo 10d, o 30 sesiones. Pérdida máxima: debit o stop.

**Qué podría generar +50%/+100%:** rally rápido del yen/FXY, expansión de FX vol y cobertura de carry.

**Frecuencia esperada:** 1–4/año.

**Datos necesarios:** FXY OHLCV/options, USDJPY spot, Treasury/JGB proxies, SPX/VIX.

**Dificultad de backtest:** media por liquidez y FX timestamps.

**Riesgos de sesgo/look-ahead:** usar datos de tipos no sincronizados; FXY no replica perfectamente USDJPY; opciones poco líquidas.

**Baselines obligatorios:** FXY shares; comprar calls cuando FXY rompe 20d sin macro filter; comprar JPY en cualquier SPX drawdown; TLT calls como proxy defensivo.

**Controles antifraude:** probar con spot USDJPY y FXY por separado; simular spreads más anchos; excluir señales durante intervención oficial si no hay timestamp fiable.

**Por qué podría fallar:** carry sigue funcionando, intervención contraria, opciones ilíquidas, o el yen deja de comportarse como hedge.

---

# 2. Prometedoras pero requieren datos especiales

## H9 — Earnings Implied-Move Underpricing

**Instrumento recomendado:** ATM straddle o 25-delta strangle con expiry inmediatamente posterior a earnings, normalmente 3–10 DTE. Mejor con liquid names: spread total <8–10%, OI >500.

**Tipo de payoff:** event volatility expansion; discrete jump convexity.

**Mecanismo económico:** no es “comprar calls antes de earnings”; es comprar varianza de evento solo cuando el implied move actual está por debajo de la distribución histórica robusta de gaps del propio ticker/sector. El edge busca mispricing de jump risk.

**Evento/señal:** earnings confirmado point-in-time; implied move = precio ATM straddle / spot; señal si implied move < percentil 35 de absolute earnings gaps históricos comparables y realized event gap dispersion sigue alta. Excluir tickers con IV extrema o spreads anchos.

**Entrada candidata:** comprar straddle/strangle al close previo al evento, distinguiendo AMC/BMO. Si earnings BMO, entrada close anterior; si AMC, entrada antes del close del día de anuncio.

**Salida candidata:** vender al primer open o primera hora post-evento. Pérdida máxima: prima.

**Qué podría generar +50%/+100%:** gap realizado >1.5–2x implied move, especialmente con gap + follow-through premarket/open.

**Frecuencia esperada:** 10–30 trades/trimestre tras filtros en universo líquido.

**Datos necesarios:** earnings calendar PIT con hora AMC/BMO, opciones, OHLCV pre/post, histórico de gaps. SEC EDGAR APIs ayudan para filings, pero no sustituyen un calendario PIT de earnings con hora fiable.

**Dificultad de backtest:** alta.

**Riesgos de sesgo/look-ahead:** calendarios de earnings backfilled; usar fecha real de filing en vez de fecha esperada operable; revisar guidance después; survivorship.

**Baselines obligatorios:** comprar todos los straddles de earnings líquidos; vender straddle; straddle con mismo debit en día no-evento; selección por IV rank simple.

**Controles antifraude:** PIT estricto; separar tickers con cambios de fecha; out-of-sample por año; capar top 1% de gaps; modelar salida al bid.

**Por qué podría fallar:** el mercado ya descuenta correctamente la cola, IV crush domina, implied move barato solo por eventos no comparables, o calendario PIT no fiable.

---

## H10 — Post-Earnings IV-Crush Convex Drift

**Instrumento recomendado:** calls/puts 21–45 DTE después de earnings, no antes. Preferencia: debit vertical 0.35–0.55 delta para reducir IV residual.

**Tipo de payoff:** event drift; gap follow-through; post-event underreaction.

**Mecanismo económico:** tras un earnings shock, el mercado puede tardar varios días/semanas en incorporar guidance, revisions y repositioning institucional. La entrada post-evento evita comprar IV máxima y busca convexidad direccional tras IV crush.

**Evento/señal:** earnings gap >8–12%; volumen >5x; close post-earnings en top 30% del rango para long, bottom 30% para short; no reversal al día siguiente. Opcional: gap rompe máximo/mínimo 6 meses.

**Entrada candidata:** al close del primer día regular post-earnings o siguiente open. Comprar call vertical en gaps positivos, put vertical en negativos, 21–45 DTE.

**Salida candidata:** +100% sobre debit, 10–20 sesiones, pérdida del gap midpoint, o cierre bajo/encima del VWAP post-earnings. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** drift de 10–30% post-evento con opción ya sin IV pre-earnings.

**Frecuencia esperada:** 5–20 señales/trimestre en universo líquido.

**Datos necesarios:** earnings PIT, OHLCV, opciones, possibly analyst revisions si se quiere segunda capa.

**Dificultad de backtest:** media-alta.

**Riesgos de sesgo/look-ahead:** identificar earnings mediante gap sin saber si era earnings; usar guidance/revisions no disponibles; survivorship.

**Baselines obligatorios:** comprar stock tras gap; opciones tras cualquier gap no-earnings; post-earnings drift sin opciones; entrada día 1 vs día 2.

**Controles antifraude:** matched sample por tamaño de gap y sector; test con y sin mega caps; costes bid/ask; embargo para no duplicar eventos del mismo ticker.

**Por qué podría fallar:** el gap ya descuenta todo, reversión de analistas, IV sigue cara, o el drift histórico desaparece.

---

## H11 — Deal-Break Put Optionality

**Instrumento recomendado:** puts o put spreads sobre targets de M&A. Contrato: 90–270 DTE, 10–30% OTM, idealmente venciendo después de hitos regulatorios. Para deals largos, roll controlado.

**Tipo de payoff:** binary downside; merger-break convexity.

**Mecanismo económico:** en deals de cash acquisition, el upside restante puede ser pequeño cuando el spread está estrecho, pero el downside si el deal fracasa puede ser 20–70%. Comprar puts convierte esa asimetría en pérdida limitada.

**Evento/señal:** cash deal anunciado; spread <2–5%; riesgo regulatorio/financiación/geopolítico no trivial; target standalone valuation muy inferior; IV no descuenta break probability. Evitar deals donde puts ya están imposibles.

**Entrada candidata:** tras compresión del spread y antes de hitos: shareholder vote, DOJ/FTC/EC decision, financing deadline. Comprar put spread hasta zona de standalone valuation.

**Salida candidata:** deal close, spread widening >2x, evento regulatorio, +100–300% sobre debit, o 30 días antes de expiry. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** deal delay severo, litigio, bloqueo regulatorio, financiación rota o buyer walk-away.

**Frecuencia esperada:** pocas señales/año; quizá 5–15 candidatos globales/año, menos con opciones líquidas.

**Datos necesarios:** base M&A PIT, terms, announcement timestamps, regulatory calendar, opciones, target/buyer OHLCV. S&P DJI y LSEG no son M&A sources, pero sus páginas muestran el tipo de publicación oficial que debe timestamp-earse para eventos de índice; M&A requiere fuente equivalente tipo press releases/filings.

**Dificultad de backtest:** alta.

**Riesgos de sesgo/look-ahead:** conocer ex post qué deals eran problemáticos; no timestamp de rumores; survivorship de deals cancelados.

**Baselines obligatorios:** comprar puts en todos los cash deals; short target stock; long put en deals con spreads aleatorios similares; comprar puts en targets sin riesgo regulatorio.

**Controles antifraude:** codificación ciega de flags regulatorios; test por periodos de política antitrust; incluir deals completados y rotos; no optimizar selección narrativa.

**Por qué podría fallar:** los puts ya incorporan el riesgo, deal se cierra rápido, spread estrecho es racional, o liquidez de opciones es nula.

---

## H12 — Index Rebalance Forced Flow

**Instrumento recomendado:** acciones o opciones 14–45 DTE sobre nombres añadidos/eliminados. Calls para additions; puts para deletions si liquidez y borrow lo permiten.

**Tipo de payoff:** forced passive flow; squeeze por demanda inelástica; event drift.

**Mecanismo económico:** fondos indexados deben ajustar posiciones alrededor de effective dates. En small/mid caps con required flow alto frente a ADV, la presión puede ser grande y predecible.

**Evento/señal:** additions/deletions Russell/S&P/Nasdaq index con required flow/ADV >10–20%; float bajo; opciones líquidas; entrada después de anuncio oficial, no antes. LSEG publica calendario y listas de Russell Reconstitution; S&P DJI mantiene media center para index announcements.

**Entrada candidata:** tras publicación oficial de listas preliminares/definitivas o anuncio S&P; comprar shares/calls si required buying alto; puts/shares short para deletions si borrow/coste viable.

**Salida candidata:** día previo al rebalance, close del effective day, +2R/+4R en shares, +100% en options, o pérdida del nivel de anuncio.

**Pérdida máxima/control:** debit en opciones; en shares, stop bajo announcement low o control por sizing; riesgo de gap.

**Qué podría generar +50%/+100%:** repricing por front-running institucional, squeeze en float bajo, IV expansion alrededor de rebalance.

**Frecuencia esperada:** concentrada en ventanas trimestrales/anuales; 10–50 candidatos brutos por rebalance, pocos con opciones.

**Datos necesarios:** index membership PIT, announcement timestamps, float, ADV, opciones, delistings.

**Dificultad de backtest:** alta si no se compra data PIT de index constituents.

**Riesgos de sesgo/look-ahead:** listas históricas corregidas; usar constituents actuales; no distinguir preliminary/final; anticipar inclusiones sin timestamp.

**Baselines obligatorios:** comprar todas las additions; sortear por required flow/ADV; shares vs options; deletions vs additions.

**Controles antifraude:** usar solo publicaciones oficiales conocidas; excluir nombres filtrados por rumores; simular entrada con delay de 1 día; out-of-sample por rebalance.

**Por qué podría fallar:** el flujo ya está arbitrado, market makers absorben, options demasiado caras, o cambios de metodología reducen edge.

---

# 3. Interesantes pero peligrosas/difíciles

## H13 — Biotech Binary Catalyst Optionality

**Instrumento recomendado:** put spreads o call spreads 30–120 DTE alrededor de PDUFA, Phase III readout o FDA panel. Nunca naked short options. Para empresas single-asset, preferir put spreads si el run-up pre-catalyst es extremo.

**Tipo de payoff:** binary event; discrete repricing; crash/approval gap.

**Mecanismo económico:** un catalyst clínico/regulatorio puede cambiar el valor fundamental en un día. La asimetría existe si la opción no refleja adecuadamente downside/upside de supervivencia, pero el mercado suele ser sofisticado y la IV puede estar muy cara.

**Evento/señal:** catalyst date PIT; empresa con dependencia de un solo activo; stock run-up >75–150% en 3–6 meses; market cap >$300M; options liquid; estructura put spread hacia cash/standalone valuation o call spread si hay underpricing real de approval.

**Entrada candidata:** 2–6 semanas antes del catalyst o tras run-up final. Put spread 20–60% OTM para negative catalyst; call spread si data científica sugiere underpricing, pero eso ya exige expertise médico.

**Salida candidata:** antes del evento si IV expansion paga +50/+100%, o inmediatamente tras catalyst. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** FDA rejection, complete response letter, failed endpoint, safety issue, o approval inesperado.

**Frecuencia esperada:** 5–20 candidatos/año con opciones razonables.

**Datos necesarios:** biotech catalyst calendar PIT, trial data, FDA dates, options, filings. SEC EDGAR puede cubrir filings, pero no sustituye una base curada de catalysts.

**Dificultad de backtest:** alta.

**Riesgos de sesgo/look-ahead:** catalyst calendars backfilled; lectura científica ex post; survivorship de biotechs quebradas; cambios de fecha.

**Baselines obligatorios:** comprar straddles en todos los catalysts; put spreads en todos los run-ups; matched non-biotech high-IV events; stock short con borrow.

**Controles antifraude:** codificación externa/ciega de catalyst; no usar lenguaje de press release posterior; separar FDA, Phase II, Phase III; capar outliers.

**Por qué podría fallar:** IV ya descuenta todo, catalyst se retrasa, resultado mixto, offering antes del evento, o spreads imposibles.

---

## H14 — Distressed Dilution Put

**Instrumento recomendado:** puts/put spreads 30–120 DTE en empresas con financiación dilutiva, going-concern risk o convertibles problemáticos. Evitar short stock salvo borrow claro.

**Tipo de payoff:** equity death spiral; crash continuation; capital-structure repricing.

**Mecanismo económico:** empresas con caja insuficiente y acceso a capital vía ATM, convertibles o toxic financing pueden entrar en espiral: emisión → caída → más emisión → pérdida de confianza. El put limita pérdida y captura colapso.

**Evento/señal:** 8-K/S-3/424B/ATM facility, going concern, covenant breach, deuda cercana, cash runway <12 meses; precio pierde soporte post-rebote; borrow caro o short interest alto.

**Entrada candidata:** después de un rebote fallido tras filing, no en el primer gap si IV explota. Put spread 30–120 DTE, long 10–20% OTM, short 50–70% OTM.

**Salida candidata:** +100–300%, nuevo financing, pérdida del mínimo de evento a favor, o 30 días antes de expiry. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** offering dilutivo, covenant/default, delisting notice, reverse split anticipation, downgrade crediticio.

**Frecuencia esperada:** varias señales/mes en small caps; pocas operables con opciones líquidas.

**Datos necesarios:** EDGAR real-time, NLP de filings, options, OHLCV, short interest/borrow si disponible. SEC EDGAR ofrece APIs de submissions y XBRL actualizadas durante el día y bulk nightly.

**Dificultad de backtest:** alta.

**Riesgos de sesgo/look-ahead:** leer filings con timestamps incorrectos; survivorship; conocer después qué financing era “toxic”; ausencia de borrow/options en nombres clave.

**Baselines obligatorios:** puts tras cualquier S-3/ATM; short shares con borrow cost; puts en distress sin filing; random small-cap puts.

**Controles antifraude:** parser automático de filings; entrada con delay de 1 sesión; incluir nombres delisted; medir borrow/options availability real.

**Por qué podría fallar:** rescue financing, squeeze, buyout, reverse split distorsiona opciones, o los puts ya están prohibitivos.

---

## H15 — Natural Gas Convex Shock

**Instrumento recomendado:** opciones sobre NG futures si tienes acceso; en retail, UNG calls/call spreads 30–60 DTE; BOIL solo para investigación táctica por decay y path dependency.

**Tipo de payoff:** commodity volatility expansion; supply/weather shock; curve squeeze.

**Mecanismo económico:** gas natural puede tener shocks de inventarios, clima, producción y storage constraints. La convexidad aparece cuando una curva deprimida pasa a estrés y la volatilidad implícita se repricing rápidamente.

**Evento/señal:** front-month NG rompe máximo 60–90d tras periodo de realized vol comprimida; storage surprise EIA alineado; curva pasa de contango a flattening/backwardation; UNG confirma con volumen >3x.

**Entrada candidata:** call spread 30–60 DTE tras confirmación de futures curve, no solo UNG. Long 0.35–0.55 delta, short 20–40% OTM. Evitar BOIL salvo salida muy corta.

**Salida candidata:** +100% sobre debit, spike de front-month >15–25%, curva se relaja, o 10–20 sesiones. Pérdida máxima: debit.

**Qué podría generar +50%/+100%:** short squeeze en futures, weather shock, storage repricing y IV expansion.

**Frecuencia esperada:** 1–4 setups/año.

**Datos necesarios:** NG futures curve, EIA storage timestamps, UNG/BOIL OHLCV/options, CME volume/OI. CME publica reports de volumen/open interest, pero datos históricos detallados de futures curve pueden requerir vendor.

**Dificultad de backtest:** alta.

**Riesgos de sesgo/look-ahead:** roll de UNG; usar continuous futures mal ajustado; storage release timestamp; clima ex post.

**Baselines obligatorios:** UNG shares; futures trend breakout; calls tras cualquier storage surprise; BOIL long con stop.

**Controles antifraude:** probar con contratos individuales no solo continuous; simular roll; separar invierno/verano; usar timestamps reales EIA.

**Por qué podría fallar:** contango/roll mata UNG, señal llega tarde, weather revierte, options caras, o BOIL decay destruye payoff.

---

# Las 5 mejores para investigar primero

## 1. H1 — Vol ETP Panic Decay Put

Mejor mezcla de **mecanismo estructural**, pérdida limitada y datos razonables. No depende de predecir el crash, sino de explotar la fase posterior de normalización. Puede generar múltiplos sobre debit y no requiere point-in-time fundamental.

## 2. H3 — Reg SHO Forced-Cover Squeeze

Tiene un mecanismo forzado verificable: fails persistentes, threshold list, restricciones de close-out/pre-borrow. Es más ruidosa y peligrosa que H1, pero puede producir colas derechas enormes con shares o calls. Buena candidata para baja frecuencia con filtros duros.

## 3. H5 — Capitulation Gap Reversal

Backtest relativamente rápido con OHLCV y delistings. Puede generar altos retornos sobre R sin necesitar opciones caras. El riesgo principal es confundir capitulación con deterioro fundamental permanente, por eso necesita filtros de exclusión.

## 4. H6 — Call-Wall Gamma Impulse

Más difícil que H5, pero con potencial de squeeze real y datos accesibles vía opciones. La clave es no inferir demasiado del option flow: usarlo como condición mecánica de positioning, no como “smart money”.

## 5. H7 — Duration Shock Convexity

Diversifica fuera de equity/QQQ. TLT options son razonablemente investigables, la señal es macroestructural y la pérdida es debit. Menos explosiva que Reg SHO, pero más limpia estadísticamente.

H2 queda como sexta: interesante para overlay de cola, pero el backtest de VIX options puede ser más delicado y puede solaparse con tu línea risk-off. H9/H10 se posponen por el bloqueo point-in-time de earnings.

---

# Plan de investigación de 2 semanas

## Día 1–2 — Data audit

Construir inventario de datos para H1, H3, H5, H6 y H7. Confirmar: cobertura OHLCV con delistings; corporate actions; opciones EOD/intraday; bid/ask; OI; expiries reales; VIX/VX; Reg SHO threshold lists; FINRA/SEC short/FTD con retrasos; TLT/yields.

Definir modelo de costes: opciones entrada ask y salida bid; alternativa mid ± 0.5 spread; shares con spread + slippage por bucket de ADV. Crear filtros mínimos de tradabilidad: precio, ADV dollar, spread, OI, volumen, market cap si disponible.

Output esperado: tabla de cobertura por año, ticker y asset class; lista de huecos; decisión de qué hipótesis puede testearse sin vendor adicional.

## Día 3–5 — Dataset y eventos

Crear event builders:

- H1: shocks VIX/VX y normalización de curva.
- H3: entrada/salida de threshold list, duración en lista, confirmación de precio.
- H5: gap-down/capitulation events con filtros de liquidez.
- H6: option volume/OI por strike, call wall breach, gamma proxy simple.
- H7: TLT range compression + yield breakout.

Cada evento debe tener timestamp, información disponible antes de entrada, instrumento elegible, contrato seleccionado, coste estimado y razón de exclusión si no opera.

Output esperado: dataset event-level con una fila por señal y contrato seleccionado.

## Día 6–8 — Screening simple

No optimizar. Usar 1–2 versiones por hipótesis:

- H1: put vertical 45/60/90 DTE.
- H3: shares con stop + calls 30/60 DTE si líquidos.
- H5: shares con stop bajo mínimo + call vertical.
- H6: shares vs call vertical 7/21 DTE.
- H7: TLT ATM option vs vertical 45/90 DTE.

Métricas primarias: retorno sobre R, P50/P75/P95/P99, hit rate, payoff ratio, max loss, expected shortfall, número de eventos, concentración por ticker/fecha, retorno sin top 1/5 trades. Métrica principal para go/no-go: **tail ratio y retorno medio capado**, no Sharpe.

Output esperado: ranking preliminar con baselines.

## Día 9–11 — Robustness

- Walk-forward temporal: train/calibration, validation, final holdout.
- Stress de costes: mid, half-spread, full-spread, adverse fill.
- Robustez de parámetros: ±20% en thresholds, entrada con delay 1 día, salida alternativa.
- Controles placebo: fechas aleatorias, tickers emparejados por liquidez/volatilidad, señales invertidas.
- Cluster control: no permitir 20 trades del mismo shock macro como observaciones independientes.

Output esperado: matriz robustez por hipótesis: edge real, edge frágil o edge explicado por outliers.

## Día 12–14 — Decisión go/no-go

Criterios de **Go**:

- al menos 50–100 eventos históricos para equity setups o 20–40 episodios para macro/vol;
- payoff positivo después de full-spread costs;
- outperformance clara frente a baseline más simple;
- cola derecha no explicada por un único trade;
- capacidad mínima razonable;
- regla implementable en live sin datos point-in-time dudosos.

Criterios de **No-Go**:

- edge desaparece con delay de 1 día;
- solo funciona en microcaps imposibles;
- depende de calendario backfilled;
- opciones no ejecutables al bid/ask real;
- un único episodio explica el resultado.

Entrega final de las 2 semanas: seleccionar 1–2 estrategias para forward paper trading con captura IBKR/Massive de quotes reales, y matar o aparcar el resto.

---

# Fuentes y notas de datos

- Massive options documentation: https://massive.com/docs/rest/options/overview
- Cboe VIX methodology: https://cdn.cboe.com/resources/indices/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf
- SEC Regulation SHO: https://www.sec.gov/investor/pubs/regsho.htm
- FINRA daily short-sale volume: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files
- NYSE trading halts: https://www.nyse.com/trade/trading-halts
- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- CME volume and open interest: https://www.cmegroup.com/market-data/volume-open-interest.html
- FTSE Russell reconstitution: https://www.lseg.com/en/ftse-russell/russell-reconstitution
- S&P DJI announcements: https://www.spglobal.com/spdji/en/media-center/news-announcements/
