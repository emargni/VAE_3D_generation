import subprocess
import os
import argparse
import shutil

def process_folder(input_dir, output_dir, resolution):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for f in os.listdir(input_dir):
        if f.lower().endswith('.obj'):
            in_path = os.path.join(input_dir, f)
            subprocess.run([r"./binvox_programs/binvox", "-d", str(resolution), in_path])
            binvox_file = os.path.splitext(f)[0] + ".binvox"
            generated = os.path.join(input_dir, binvox_file)
            if os.path.exists(generated):
                shutil.move(generated, os.path.join(output_dir, binvox_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--resolution", type=int, default=10)
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder, args.resolution)
