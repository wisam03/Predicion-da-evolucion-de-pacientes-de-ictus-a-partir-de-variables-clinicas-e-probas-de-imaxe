# Predicion da evolucion de pacientes de ictus a partires de variables clinicas e probas de imaxe

Repositorio do Traballo de Fin de Grao que contén os scripts e cadernos jupyter necesarios para reproducir os experimentos de segmentación de lesión isquémica, extracción de características radiómicas e modelado preditivo do resultado funcional (escala mRS) en pacientes con ictus isquémico agudo.

## Contido do repositorio

| Ficheiro | Descrición |
|---|---|
| `segmentacion_dagmnet.py` | Prepara as imaxes e lanza a segmentación automática con DAGMNet |
| `extraccion_radiomicas.py` | Extrae características radiómicas das máscaras de lesión e actualiza o Excel |
| `TOAST_R_A_NIH_A_vasc.ipynb` | Modelado con variables clínicas e características vasculares |
| `R_A_VOL_TOAST_shape_volume_ml_intensity_range.ipynb` | Modelado con variables clínicas e radiómicas de forma e de intensidade|

---

## Dependencias e instalación

### 1. Instalar DAGMNet (repositorio externo)

```bash
git clone https://github.com/Chin-Fu-Liu/Acute-stroke_Detection_Segmentation/
```

Descarga os pesos preentrenados desde:
https://www.nitrc.org/frs/?group_id=1520

Seleccionar a versión `ADSv1.zip` e descomprimila. Os ficheiros `.h5` atópanse en `Acute-stroke_Detection_Segmentation/data/Trained_Nets/`; copialos ó mesmo directorio da instalación local.

### 3. Instalar dependencias

```bash
# Dependencias de DAGMNet (versións exactas obrigatorias)
pip install numpy==1.19.5 nibabel==3.2.1 scipy==1.4.1
pip install scikit-image==0.18.1 scikit-learn==0.24.1
pip install dipy==1.4.0
pip install tensorflow==2.0.0
pip install h5py==2.10.0   # versión crítica, non cambiar

# Dependencias adicionais dos scripts propios
pip install pandas openpyxl
pip install jupyter matplotlib seaborn
```

---

---

## Uso: fluxo de traballo completo

### Paso 1 — Segmentación automática das lesións

```bash
python segmentacion_dagmnet.py \
    /ruta/Acute-stroke_Detection_Segmentation \
    /datos/dwi \
    /datos/adc \
    /resultados/segmentacion

# Para un único paciente:
python segmentacion_dagmnet.py ... --paciente ST_FIDIS_001
```

**Formato esperado dos ficheiros de entrada:**
- Os ficheiros DWI/b0 deben estar en `/datos/dwi/` e os ADC en `/datos/adc/`
- Os nomes deben incluír o patrón `ST_FIDIS_N` (p. ex. `ST_FIDIS_001_dwi.nii.gz`)
- A busca é recursiva: non é necesario ningún nivel de organización en subcarpetas

**Saída por paciente** (en `/resultados/segmentacion/_ADS_input/ST_FIDIS_N/`):
ST_FIDIS_001_DAGMNet_CH3_Lesion_Predict.nii.gz
ST_FIDIS_001_DAGMNet_CH3_Lesion_Predict_MNI.nii.gz
ST_FIDIS_001_DAGMNet_CH3_Lesion_Predict_result.png
ST_FIDIS_001_ADC.nii.gz
ST_FIDIS_001_ADC_MNI.nii.gz
ST_FIDIS_001_b0.nii.gz
ST_FIDIS_001_b0_MNI.nii.gz
ST_FIDIS_001_DWI.nii.gz
ST_FIDIS_001_DWI_MNI.nii.gz
ST_FIDIS_001_DWI_Norm_MNI.nii.gz
ST_FIDIS_001_volume_brain_regions.txt
ST_FIDIS_001_result.png

### Paso 2 — Extracción de características radiómicas

```bash
python extraccion_radiomicas.py \
    --excel /ruta/Excel.xlsx \
    --root  /resultados/segmentacion/_ADS_input \
    --prefix ST_
```

O script actualiza o Excel de forma incremental: as celas xa cubertas nunca se sobrescriben. Pódese executar múltiples veces con seguridade.

**Características extraídas por paciente:**
- **Morfolóxicas** (8): volume en mL, área superficial, esfericidade, bounding box, extent, compactness, compoñentes conexas
- **Intensidade ADC** (14): media, varianza, std, mín, máx, rango, mediana, p10, p90, IQR, asimetría, curtose, enerxía, entropía
- **Textura GLCM** (6): contraste, disimilaridade, homoxeneidade, enerxía, correlación, ASM
- **Rexionais atlas JHU** (26 rexións × 4 columnas): vóxeles, mL, % afectado, indicador binario
- **Territorios vasculares** (ACA/MCA/PCA/VB × 3): vóxeles, mL, % do total da lesión
- **Globais**: volume intracraneal (mL), % da lesión respecto ao ICV

### Paso 3 — Modelado preditivo

```bash
jupyter notebook
```

Abre o caderno correspondente ao momento temporal:

| Caderno | Momento | Variables principais |
|---|---|---|
| `TOAST_R_A_NIH_A_vasc.ipynb` | Ingreso / 4-10 día / Alta | NIHSS, TOAST, R_A, vasculares |
| `R_A_VOL_TOAST_shape_volume_ml_intensity_range.ipynb` | Ingreso / 4-10 día / Alta | NIHSS, VOL, radiómicas morfolóxicas e de intensidade |

Os cadernos realizan automaticamente validación cruzada estratificada (5 particións), cálculo de métricas (F1-macro, kappa, MAE, Acc±1, ρ de Spearman) e selección do mellor modelo.

---

## Notas técnicas importantes

- **h5py==2.10.0** é unha versión crítica. Calquera outra versión provocará erros ao cargar os modelos de DAGMNet.
- **TOAST** só está dispoñible entre os días 4 e 10 tras o ingreso.
- **Características radiómicas** de imaxe están dispoñibles desde o ingreso (derívanse da DWI basal).
- O volume en mL da lesión calcúlase a partir dos vóxeles da máscara MNI en espazo de $1\times1\times1$ mm³ divididos por 1000.
- O modelo de segmentación empregado é `DAGMNet_CH3` (3 canles: DWI + b0 + ADC).

---

## Referencia de DAGMNet

Liu CF, Hsu J, Xu X, et al. Deep learning-based detection and segmentation of diffusion abnormalities in acute ischemic stroke. *Commun Med* **1**, 61 (2021). https://doi.org/10.1038/s43856-021-00062-8

---
