import tensorflow as tf
import os

ruta_base = r'/home/chayote/Datasets/SpaceNet.FLARE.imam_alam' 
ruta_cuarentena = r'/home/chayote/Datasets/corruptos' 
archivos_movidos = 0

if not os.path.exists(ruta_cuarentena):
    os.makedirs(ruta_cuarentena)
    print(f"Creada carpeta de cuarentena en: {ruta_cuarentena}")

print(f"Iniciando REVISIÓN SEGURA con TensorFlow en: {ruta_base}")
print("Esto puede tardar, ya que se está leyendo cada imagen...")

for nombre_clase in os.listdir(ruta_base):
    ruta_clase = os.path.join(ruta_base, nombre_clase)
    
    if os.path.isdir(ruta_clase):
        print(f"\n--- Revisando Carpeta: {nombre_clase} ---")
        archivos_en_clase = os.listdir(ruta_clase)
        
        for i, nombre_archivo in enumerate(archivos_en_clase):
            ruta_archivo = os.path.join(ruta_clase, nombre_archivo)
            
            if (i + 1) % 500 == 0:
                print(f"  ... revisando {i+1}/{len(archivos_en_clase)}")

            if os.path.isdir(ruta_archivo):
                continue

            try:
                if os.path.getsize(ruta_archivo) == 0:
                    raise tf.errors.InvalidArgumentError(None, None, "El archivo está vacío (0 bytes)")
                
                raw_bytes = tf.io.read_file(ruta_archivo)
                
                tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)

            except tf.errors.InvalidArgumentError as e:
                print(f"  [MOVIDO] Archivo problemático: {nombre_archivo}")
                print(f"     -> Razón: {e.message.splitlines()[0]}...") 
                
                try:
                    destino_final = os.path.join(ruta_cuarentena, f"{nombre_clase}_{nombre_archivo}")
                    os.rename(ruta_archivo, destino_final)
                    archivos_movidos += 1
                except Exception as move_e:
                    print(f"  [ERROR GRAVE] No se pudo mover {nombre_archivo}: {move_e}")
            
            except Exception as other_e:
                print(f"  [IGNORADO] Error no relacionado al leer {nombre_archivo}: {other_e}")


print(f"\n¡Revisión segura completada! Se movieron {archivos_movidos} archivos a '{ruta_cuarentena}'.")