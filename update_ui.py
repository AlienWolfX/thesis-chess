import os
import subprocess

def checkRemote():
    result = subprocess.run(["git", "diff", "--name-only", "origin/main"], capture_output=True, text=True)
    modified_files = result.stdout.splitlines()
    remoteFile = [f for f in modified_files if f.startswith('ui/') and f.endswith('.ui')]
    return remoteFile

def checkLocal():
    local_files = [os.path.join('ui', f) for f in os.listdir('ui') if f.endswith('.ui')]
    return local_files

def newFile(localFile):
    repo = subprocess.run(["git", "ls-files", "ui/*.ui"], capture_output=True, text=True)
    repo_files = repo.stdout.splitlines()
    new = [f for f in localFile if f not in repo_files]
    return new

def getTime(file_path):
    return os.path.getmtime(file_path)

def main():
    remoteFile = checkRemote()
    localFile = checkLocal()
    new = newFile(localFile)
    
    if not new:
        print("No new .ui files found.")
    else: 
        print("New local .ui files:", new)
    
    combinedUi = set(remoteFile + new)
    
    if not combinedUi:
        print("No .ui files found.")
        return
    
    print("The following .ui files have been modified:")
    for ui_file in combinedUi:
        print(f"- {ui_file}")
    
    proceed = input("Do you want to update the corresponding .py files? (yes/no): ").strip().lower()
    
    if proceed not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    for ui_file in combinedUi:
        py_file = f"ui_{os.path.splitext(os.path.basename(ui_file))[0]}.py"
        
        if not os.path.exists(py_file) or getTime(ui_file) > getTime(py_file):
            print(f"Updating {py_file} from {ui_file}")
            subprocess.run(["pyuic6", "-o", py_file, ui_file])
        else:
            print(f"{py_file} is up to date")

if __name__ == "__main__":
    main()