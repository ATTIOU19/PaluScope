# Règles de travail

## Branches
- `main` : code stable uniquement.
- Une branche par tâche importante :
  - `feature/phase0-data-exploration`
  - `feature/phase0-quality-analyzer`
  - `feature/phase0-candidate-detection`
  - `fix/phase0-thresholding`
  - `docs/phase0-readme`

## Règles
1. Pas de commit direct sur `main`.
2. Toute fusion vers `main` (Pull Request) est relue par au moins un autre membre.
3. On ne modifie pas le travail d'un autre membre sans coordination.
4. Les notebooks servent à explorer ; le code réutilisable va dans `src/`.
5. Pas de gros fichiers de dataset dans Git.
6. Les décisions importantes sont consignées dans `docs/decisions.md`.
7. Idée hors périmètre → `docs/a_etudier_plus_tard.md`, pas d'implémentation immédiate.

## Cycle d'une tâche
À faire → En développement → Test → Review → Intégrée

## Definition of Done
- Le code fonctionne sur un exemple ou un petit jeu de test
- Le fonctionnement a été expliqué à au moins un autre membre
- Le résultat est visualisé ou vérifié
- Les hypothèses importantes sont documentées
- Le code respecte la structure du projet
- Une revue a été faite
- Intégrable sans casser les autres composants

## Messages de commit
Courts et au présent : `add blur measure`, `fix hsv conversion`, `docs: update readme`.
