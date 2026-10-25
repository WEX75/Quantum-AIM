# Git în echipă, fără merge conflicts

## Regulile (le respectăm toată ziua)
1. **`main` e mereu verde.** Nimeni nu face commit direct pe `main`.
2. **Branch-uri scurte**, de 1–2 ore: `darius/noise-aware`, `ana/slides`. Un branch = o idee.
3. **Fiecare are fișierele lui.** Conflictele apar doar când doi oameni editează aceleași linii.
   | Cine | Fișiere |
   |---|---|
   | Owner de cod | `qai/`, `qhack/`, `tests/` |
   | Owner de poveste | `docs/PITCH.md`, `README.md`, slide-uri, `docs/KIT.md` |
   | Toată lumea | `docs/LOG.md` — doar adaugi linii la final, git le păstrează pe ale tuturor automat |
4. **Experimente noi = fișiere noi** în `qai/experiments/exp_<topic>.py` (copiezi `exp_template.py`).
   Dacă doi lucrează pe experimente diferite, nu ating niciodată același fișier.
5. **`results/` nu intră în git**, deci JSON-urile și graficele nu fac conflicte. Plot-urile finale
   se copiază în `docs/` doar la final.
6. **Sincronizezi des:** `make sync` la începutul oricărei sarcini noi.

## Fluxul pentru fiecare idee
```bash
git checkout main && git pull            # pornești de la ultima versiune
git checkout -b darius/noise-aware       # branch nou
# ... lucrezi ...
git add -A && git commit -m "qai: noise-aware training sweep"
make merge                               # teste + rebase pe main + PR + squash merge + înapoi pe main
```
`make merge` rulează testele rapide, aduce ultimul `main` sub branch-ul tău, urcă branch-ul,
deschide un PR și îl îmbină. Dacă testele pică, se oprește și nu strică `main`.

## Prima configurare (o dată, în seara asta)
Darius creează repo-ul și îl urcă:
```bash
gh repo create breaq-2026 --public --source=. --push
gh repo edit --enable-squash-merge --delete-branch-on-merge
```
Colegii: Darius îi adaugă (`gh repo edit` sau Settings → Collaborators), apoi ei:
```bash
gh auth login
gh repo clone <user>/breaq-2026 && cd breaq-2026
python3 -m venv .venv && source .venv/bin/activate && make install && make test
```

## Dacă totuși apare un conflict
```bash
git status                     # vezi fișierele cu conflict
```
Editezi fișierul păstrând intenția ambelor părți, ștergi markerii `<<<<<<<`, `=======`, `>>>>>>>`,
rulezi `make test-fast`, apoi `git add <fișier> && git rebase --continue`.

Ai panicat? `git rebase --abort` te întoarce exact unde erai înainte.

## Două sesiuni de lucru în paralel pe același laptop
Ca să nu se calce pe fișiere, fiecare sesiune lucrează în propriul folder (worktree):
```bash
git worktree add ../breaq-exp2 -b darius/exp2
cd ../breaq-exp2
```
Când termini: `make merge`, apoi `git worktree remove ../breaq-exp2`.
