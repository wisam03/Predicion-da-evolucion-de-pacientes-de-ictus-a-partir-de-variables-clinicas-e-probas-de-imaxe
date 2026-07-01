#!/usr/bin/env python3

# Importación de las librerías necesarias
import os
import sys
import subprocess
import numpy as np
import nibabel as nib

def buscar_pacientes_recursivo(directorio, palabra_clave):
    """
    Función que busca archivos NIfTI (.nii o .nii.gz) en carpetas y subcarpetas.
    Filtra por una palabra clave (como 'dwi' o 'adc') y organiza los
    resultados en un diccionario usando el ID del paciente como clave.
    """
    mapa = {}
    for raiz, _, archivos in os.walk(directorio):
        for archivo in archivos:
            # Compruebación de si el archivo coincide con la palabra clave y es formato NIfTI
            if palabra_clave.lower() in archivo.lower() and archivo.endswith((".nii", ".nii.gz")):
                import re
                # Búsqueda del patrón estándar 'ST_FIDIS_' seguido de números en el nombre del archivo
                match = re.search(r'(ST_FIDIS_\d+)', archivo)
                if match:
                    paciente_id = match.group(1)
                    # Se guarda la ruta completa asociada a ese ID único de paciente
                    mapa[paciente_id] = os.path.join(raiz, archivo)
    return mapa

def reconstruir_b1000(b0_path, adc_path, out_path, b_value=1000):
    """
    Función que calcula de forma matemática una imagen b1000 artificial (DWI sintética)
    combinando la imagen base b0 y el mapa ADC con la fórmula física de difusión.
    """
    b0_img = nib.load(b0_path)
    b0_data = b0_img.get_fdata()
    adc_data = nib.load(adc_path).get_fdata()

    # Si la imagen b0 tiene 4 dimensiones (tiempo o secuencias), nos quedamos solo con el primer volumen (3D)
    if b0_data.ndim == 4:
        print(f" Detectado archivo 4D, extrayendo primer volumen para b0.")
        b0_data = b0_data[..., 0]

    # Corrección de escala para el ADC: si los valores son muy altos,
    # se dividen para adaptarlos a la escala que espera la fórmula matemática.
    if np.max(adc_data) > 10:
        adc_data = adc_data / 1_000_000.0   # Se pasa de micrómetros cuadrados a milímetros cuadrados por segundo
    elif np.max(adc_data) > 0.01:
        adc_data = adc_data / 1000.0    # Se ajusta otra escala común de adquisición de la máquina

    # Se aplica la fórmula física: b1000 = b0 * e^(-b * ADC)
    b1000_data = b0_data * np.exp(-b_value * adc_data)
    # Se reemplaza cualquier valor negativo erróneo por cero absoluto
    b1000_data = np.clip(b1000_data, 0, None)

    # Se copia la información técnica (cabecera) del b0 original para que la nueva imagen mantenga el mismo formato
    header = b0_img.header.copy()
    header.set_data_shape(b1000_data.shape)
    header.set_data_dtype(np.float32)

    # Creación del nuevo archivo médico NIfTI con los datos calculados y se almacena en disco
    b1000_nifti = nib.Nifti1Image(
        b1000_data.astype(np.float32),
        b0_img.affine,   # Se mantiene la orientación y posición en el espacio del b0
        header
    )
    nib.save(b1000_nifti, out_path)
    return out_path

def preparar_input_dagmnet(paciente_id, ruta_b0, ruta_adc, ads_input_dir):
    """
    Función que crea la estructura de carpetas específica que necesita la red neuronal DAGMNet
    y guarda los 3 archivos requeridos (ADC, b0 y DWI) asegurando que todos sean estrictamente 3D.
    """
    pac_dir = os.path.join(ads_input_dir, paciente_id)
    os.makedirs(pac_dir, exist_ok=True)

    # Se copia y se asegura que el mapa ADC sea 3D (se eliminan dimensiones extra si las hay)
    img_adc = nib.load(ruta_adc)
    data_adc = img_adc.get_fdata()
    if data_adc.ndim >= 4: data_adc = data_adc[..., 0]
    nib.save(nib.Nifti1Image(data_adc.astype(np.float32), img_adc.affine[:4,:4]),
             os.path.join(pac_dir, f"{paciente_id}_ADC.nii.gz"))

    # Se copia y se asegura que el archivo b0 sea 3D para evitar fallos de lectura en el modelo
    img_b0 = nib.load(ruta_b0)
    data_b0 = img_b0.get_fdata()
    if data_b0.ndim >= 4: data_b0 = data_b0[..., 0]
    nib.save(nib.Nifti1Image(data_b0.astype(np.float32), img_b0.affine[:4,:4]),
             os.path.join(pac_dir, f"{paciente_id}_b0.nii.gz"))

    # Se genera la tercera imagen obligatoria (b1000/DWI) usando la función matemática anterior
    ruta_b1000 = os.path.join(pac_dir, f"{paciente_id}_DWI.nii.gz")
    reconstruir_b1000(ruta_b0, ruta_adc, ruta_b1000)

    return pac_dir

def ejecutar_dagmnet(dagmnet_dir, paciente_dir, model="DAGMNet_CH3"):
    """
    Función que llama y ejecuta el script externo 'ADSRun.py' de la red neuronal pasando la carpeta
    del paciente como parámetro para que realice la segmentación automática.
    """
    ads_script = os.path.join(dagmnet_dir, "codes", "ADSRun.py")
    cmd = ["python", ads_script, "-input", paciente_dir, "-model", model]
    print(f"Ejecutando: {' '.join(cmd)}")

    # Se lanza el proceso en el sistema y se espera a que termine para devolver si fue exitoso (código 0) o falló
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    # Se verifica que el usuario haya pasado todos los directorios obligatorios por terminal
    if len(sys.argv) < 5:
        print("Uso: python script.py <dagmnet_dir> <dwi_dir> <adc_dir> <output_dir> [--paciente ID]")
        sys.exit(1)

    # Se capturan las rutas de las carpetas introducidas por el usuario
    dag_path, b0_dir, adc_dir, out_path = sys.argv[1:5]
    # Si el usuario usó la bandera opcional '--paciente', se procesará únicamente ese ID
    filtro = sys.argv[sys.argv.index("--paciente") + 1] if "--paciente" in sys.argv else None

    # Se crea la carpeta temporal de entrada general para almacenar los datos organizados por paciente
    ads_input_dir = os.path.join(out_path, "_ADS_input")
    os.makedirs(ads_input_dir, exist_ok=True)

    # Se localizan y mapean de forma independiente todas las imágenes b0 y mapas ADC disponibles
    mapa_b0 = buscar_pacientes_recursivo(b0_dir, "dwi")
    mapa_adc = buscar_pacientes_recursivo(adc_dir, "adc")

    # Se emparejan los pacientes: solo se procesarán aquellos que tengan ambos archivos (b0 y ADC)
    pacientes = sorted(set(mapa_b0.keys()) & set(mapa_adc.keys()))

    print(f"Indexados {len(pacientes)} pacientes.")

    # Bucle principal para procesar uno a uno a los pacientes seleccionados
    for p_id in pacientes:
        # Si se especificó un paciente concreto con '--paciente' y no es este, se salta
        if filtro and p_id != filtro: continue

        # Paso 1: Se organizan los archivos del paciente y se calcula la secuencia b1000 faltante
        print(f"\n{'='*40}\nProcesando: {p_id}")
        pac_dir = preparar_input_dagmnet(p_id, mapa_b0[p_id], mapa_adc[p_id], ads_input_dir)

        # Paso 2: Se envían los archivos listos a la red neuronal para obtener el resultado
        ret = ejecutar_dagmnet(dag_path, pac_dir)
        print(f"Resultado: {'OK' if ret == 0 else 'ERROR'}")

if __name__ == "__main__":
    main()
