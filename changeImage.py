import shutil

def duplicate_and_replace(original_file, new_file):
    try:
        shutil.copy2(original_file, new_file)  # Copy the original file to new file
        print(f"{original_file} duplicated and replaced {new_file} successfully.")
    except FileNotFoundError:
        print("One or both of the files does not exist.")


