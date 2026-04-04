# Training Presets

Preset dedicati esclusivamente al caso d'uso drone + marker stampato (`fire.png`).

Il configuratore usa due famiglie:

- `configs/presets/dataset/`
- `configs/presets/training/`

Ogni preset contiene il campo top-level `description`, mostrato nella tab `Presets` del configuratore.

## Matrice dataset essenziale

Assi coperti:

- quota drone: `basso` / `alto`
- sorgente sfondi: `unsplash` / `sintetiche geometriche`

Preset disponibili:

1. `configs/presets/dataset/drone-basso-unsplash.yaml`
2. `configs/presets/dataset/drone-alto-unsplash.yaml`
3. `configs/presets/dataset/drone-basso-sintetiche.yaml`
4. `configs/presets/dataset/drone-alto-sintetiche.yaml`

## Preset training nano (modalita' operative)

1. `configs/presets/training/nano-precisione-ricerca.yaml`
2. `configs/presets/training/nano-bilanciato-operativo.yaml`
3. `configs/presets/training/nano-close-pass-conferma.yaml`

## Combinazioni consigliate rapide

1. Ricerca con FP minimi: `drone-alto-unsplash` + `nano-precisione-ricerca`
2. Operativo generale: `drone-basso-unsplash` + `nano-bilanciato-operativo`
3. Conferma ravvicinata: `drone-basso-unsplash` + `nano-close-pass-conferma`
4. Nessun dato reale disponibile: `drone-basso-sintetiche` + `nano-precisione-ricerca`