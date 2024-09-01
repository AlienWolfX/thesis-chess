# Script to batch update modified .ui files
import os
import subprocess

def checkRemote():
    result = subprocess.run(["git", "diff", "--name-only", "origin/main"], capture_output=True, text=True)
    modified_files = result.stdout.splitlines()
    ui_files = [f for f in modified_files if f.endswith('.ui')]
    return ui_files

def getTime(file_path):
    return os.path.getmtime(file_path)

def main():
    ui_files = checkRemote()
    
    if not ui_files:
        print("No modified .ui files found.")
        return
    
    print("The following .ui files have been modified:")
    for ui_file in ui_files:
        print(f"- {ui_file}")
    
    proceed = input("Do you want to update the corresponding .py files? (yes/no): ").strip().lower()
    
    if proceed not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    for ui_file in ui_files:
        py_file = f"ui_{os.path.splitext(ui_file)[0]}.py"
        
        if not os.path.exists(py_file) or getTime(ui_file) > getTime(py_file):
            print(f"Updating {py_file} from {ui_file}")
            subprocess.run(["pyuic6", "-o", py_file, ui_file])
        else:
            print(f"{py_file} is up to date")

if __name__ == "__main__":
    main()