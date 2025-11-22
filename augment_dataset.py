import os
import random
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

BASE_PATH = '/home/chayote/Datasets/SpaceNet.FLARE.imam_alam'
TARGET_COUNT = 6000
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
AUG_PER_IMAGE = 3


def augment_image(args):
    """Aplica aumentos simples a una imagen."""
    img_path, output_dir, idx = args
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return 0

    angle = random.choice([0, 90, 180, 270, random.randint(-25, 25)])
    img = img.rotate(angle, expand=True)

    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.3:
        img = ImageOps.flip(img)

    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

    new_name = f"aug_{idx}_{random.randint(10000,99999)}.jpg"
    img.save(os.path.join(output_dir, new_name), quality=90)
    return 1


def process_class_folder(class_folder):
    """Equilibra una carpeta de clase a TARGET_COUNT imágenes."""
    target_dir = os.path.join(BASE_PATH, class_folder)
    all_images = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.lower().endswith(IMAGE_EXT)
    ]

    count = len(all_images)
    print(f"\n🔹 {class_folder}: {count} imágenes actuales")


    if count < TARGET_COUNT:
        to_generate = TARGET_COUNT - count
        print(f"  + Generando {to_generate} imágenes nuevas...")

        jobs = []
        for i in range(to_generate):
            src = random.choice(all_images)
            jobs.append((src, target_dir, i))

        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            for _ in tqdm(pool.imap_unordered(augment_image, jobs), total=len(jobs)):
                pass

    # Verificar conteo final
    final_count = len([
        f for f in os.listdir(target_dir)
        if f.lower().endswith(IMAGE_EXT)
    ])
    print(f"  <3 Total final: {final_count}")
    return class_folder, final_count


def main():
    class_folders = [
        'constellation', 'comet', 'star',
        'black hole', 'asteroid', 'nebula',
        'planet', 'galaxy'
    ]

    summary = {}
    for folder in class_folders:
        folder_path = os.path.join(BASE_PATH, folder)
        if not os.path.exists(folder_path):
            print(f"! Carpeta no encontrada: {folder_path}")
            continue
        name, count = process_class_folder(folder)
        summary[name] = count

    print("\n Resumen final por clase:")
    for k, v in summary.items():
        print(f"  {k:15s}: {v} imágenes")


if __name__ == "__main__":
    main()
