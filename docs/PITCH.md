# Prezentarea — construită în paralel cu codul

**Regula:** povestea se scrie *înainte* de rezultate, cu goluri. Fiecare experiment umple un gol.
Dacă un experiment nu umple niciun gol din pitch, nu îl facem.

## Cum lucrăm în paralel
| Când | Cod | Prezentare |
|---|---|---|
| 13:00 | Alegem ideea | Scriem titlurile slide-urilor ca **afirmații cu goluri** (vezi mai jos) |
| 14:00–20:00 | Bucle `/exp` | După fiecare `/exp`, numărul nou intră în tabelul „Cifre” de aici |
| 21:00 | Experimentul complet | Prima versiune de slide-uri (Claude Slides în chat, se exportă PPTX/PDF) |
| 02:00 | Plot-uri finale | Textul vorbit, cronometrat o dată |
| Dum. 10:00 | `/ship` | Slide-uri finale + repetiție ×2 + Q&A |

Cine face ce: unul e „owner de cod”, celălalt „owner de poveste” (slide-uri, text, grafice frumoase,
întrebări de juriu). Vă sincronizați la fiecare 2h, 10 minute.

## Cifre (se completează pe parcurs)
| Afirmație | Cifra | Sursa (comandă / fișier) |
|---|---|---|
| Acuratețea scade de la __ la __ sub zgomot realist | | `python run.py summary` |
| Metoda noastră păstrează __% la zgomot p=__ | | |
| Baseline clasic: __% cu __ parametri; noi: __% cu __ parametri | | |
| Entanglementul contribuie cu __ (sau nu contribuie) | | |
| Cost: __ s antrenare, __ shots | | |

## Structura (5 min; varianta de 3 min = slide-urile 1, 3, 4, 5, 7)
1. **Hook (15s).** O cifră care surprinde. *„Același model cuantic: 100% pe simulator perfect, 51% cu zgomotul unui calculator real.”*
2. **Problema (30s).** De ce zgomotul strică QML pe hardware-ul de azi. Fără formule, cu o imagine.
3. **Ideea noastră (45s).** O propoziție + diagrama pipeline-ului (date → encoding → circuit → zgomot → mitigare → predicție).
4. **Rezultatul principal (60s).** UN grafic. Spui ce să privească: „linia portocalie stă sus, cea albastră cade”.
5. **De ce să ne credeți (45s).** Baseline clasic, nr. de parametri, 3 seed-uri, ce NU funcționează. Onestitatea bate hype-ul.
6. **Ce am mai aflat (45s).** Cele 2 analize ieftine (ex: entanglementul, tipul de zgomot), câte o frază fiecare.
7. **Demo / ce urmează (30s).** Dashboard live sau „pe hardware real ne așteptăm la…”. Final cu propoziția de la slide-ul 3.

## Întrebări probabile de la juriu (pregătiți răspunsuri de 20s)
- E un avantaj cuantic real sau doar un clasificator mic? → param count + baseline
- De ce acest encoding? → justificare + ablație
- Cât de realist e modelul de zgomot? → ce canale, ce valori, de unde
- Ar merge pe hardware real? Câte shots? → shot budget, adâncime, nr. CNOT
- Ce rol joacă entanglementul? → ablația ring/none
- Ce ați face cu încă o săptămână?

## Reguli pentru slide-uri
- O afirmație pe slide, în titlu. Textul de pe slide ≤ 15 cuvinte; restul îl spui.
- Graficele: axe mari, legendă clară, culoarea „noastră” mereu aceeași.
- Fiecare cifră de pe slide există în tabelul „Cifre” și se poate reproduce cu o comandă.
- Ultimul slide: link la repo (QR code).
