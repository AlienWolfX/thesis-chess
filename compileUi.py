import os
import subprocess
from pathlib import Path

def find_ui_files(ui_dir="ui"):
    """Find all .ui files in the specified directory"""
    ui_files = []
    if os.path.exists(ui_dir) and os.path.isdir(ui_dir):
        for file in os.listdir(ui_dir):
            if file.endswith(".ui"):
                ui_files.append(file)
    return ui_files

def compile_ui_file(input_path, output_dir="forms"):
    """Compile a single .ui file to .py using pyuic6"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"{Path(file_name).stem}.py")
    
    try:
        subprocess.run(["pyuic6", input_path, "-o", output_path], check=True)
        print(f"Successfully compiled {file_name} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {file_name}: {e}")
    except FileNotFoundError:
        print("pyuic6 not found. Please ensure PyQt6 is installed.")

def main():
    ui_files = find_ui_files()
    
    if not ui_files:
        print("No .ui files found in the 'ui' directory.")
        return
    
    print("Found the following .ui files:")
    for i, file in enumerate(ui_files, 1):
        print(f"{i}. {file}")
    
    while True:
        print("\nOptions:")
        print("1. Compile a specific file")
        print("2. Compile all files")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            try:
                file_num = int(input(f"Enter file number to compile (1-{len(ui_files)}: "))
                if 1 <= file_num <= len(ui_files):
                    selected_file = ui_files[file_num - 1]
                    compile_ui_file(os.path.join("ui", selected_file))
                else:
                    print("Invalid file number.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == "2":
            print("Compiling all .ui files...")
            for file in ui_files:
                compile_ui_file(os.path.join("ui", file))
            print("All files compiled.")
        
        elif choice == "3":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()