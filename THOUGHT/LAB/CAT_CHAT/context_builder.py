import os
import sys

def scan_folder_structure(folder_path, indent_level=0):
    """
    Recursively scans the folder and returns a list of strings representing the file architecture.
    Skips folders that start with a dot (hidden folders).
    """
    result = []
    try:
        items = sorted(os.listdir(folder_path))
    except OSError as e:
        print(f"Error accessing {folder_path}: {e}")
        return []

    for item in items:
        if item.startswith('.'):
            continue
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            result.append(f"{'  ' * indent_level}- **{item}/**")
            result.extend(scan_folder_structure(item_path, indent_level + 1))
        else:
            result.append(f"{'  ' * indent_level}- {item}")
    return result

def write_structure_to_md(output_path, folder_structure):
    """Writes the folder structure to a Markdown file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write("# Folder Structure\n\n")
            md_file.write('\n'.join(folder_structure))
        print(f"Folder structure written to {output_path}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}")

def combine_all_md_files_recursive(root_folder, output_folder):
    """
    Recursively finds ALL MD files in the root_folder and combines them into ONE master file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    master_content = []
    master_content.append(f"# Master Combined Document\n")
    master_content.append(f"Source: {root_folder}\n\n")

    file_count = 0

    for root, dirs, files in os.walk(root_folder):
        # Skip hidden folders and the output folder itself
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if os.path.abspath(root) == os.path.abspath(output_folder):
            continue

        for file in sorted(files):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                
                # Add explicit start tag
                master_content.append(f"START OF FILE: {file}\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        master_content.append(f.read())
                    
                    # Add explicit end tag and separator
                    master_content.append(f"\nEND OF FILE: {file}\n")
                    master_content.append("\n---\n")
                    file_count += 1
                except IOError as e:
                    print(f"Error reading {file_path}: {e}")

    if file_count == 0:
        print("No Markdown files found to combine.")
        return

    # Write Master MD File
    folder_name = os.path.basename(os.path.normpath(root_folder))
    output_md_path = os.path.join(output_folder, f"{folder_name}_FULL_COMBINED.md")
    try:
        with open(output_md_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(master_content))
        print(f"Created Master Combined Markdown: {output_md_path} ({file_count} files)")
    except IOError as e:
        print(f"Error writing {output_md_path}: {e}")

    # Convert to TXT
    output_txt_path = os.path.join(output_folder, f"{folder_name}_FULL_COMBINED.txt")
    md_to_txt(output_md_path, output_txt_path)


def md_to_txt(md_file_path, txt_file_path):
    """Converts a Markdown file to a plain text file by removing Markdown formatting."""
    try:
        with open(md_file_path, 'r', encoding='utf-8') as md_file:
            lines = md_file.readlines()
    except IOError as e:
        print(f"Error reading {md_file_path}: {e}")
        return

    title = os.path.splitext(os.path.basename(md_file_path))[0]
    text_lines = [f"{title}\n\n"]

    for line in lines:
        if line.strip().startswith('#'):
            continue
        if line.strip() == '---':
            continue
        text_lines.append(line)

    try:
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.writelines(text_lines)
        print(f"Converted to Text: {txt_file_path}")
    except IOError as e:
        print(f"Error writing {txt_file_path}: {e}")


def main():
    print("--- Context Builder System ---")
    print("1. Generate Folder Structure Map (Markdown)")
    print("2. Combine ALL MD files into ONE Master File (MD & TXT)")
    print("3. Run ALL (Map -> Combine)")
    print("q. Quit")

    choice = input("\nEnter your choice: ").strip().lower()

    if choice == 'q':
        return

    # Option 1: Map
    if choice in ['1', '3']:
        folder_path = input("Enter folder to scan for structure: ").strip()
        
        if os.path.exists(folder_path):
            folder_name = os.path.basename(folder_path)
            output_folder = os.path.join(folder_path, f"{folder_name} COMBINED")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            

            output_path = os.path.join(output_folder, f"{folder_name}_FULL_STRUCTURE.md")
            
            structure = scan_folder_structure(folder_path)
            write_structure_to_md(output_path, structure)
        else:
            print("Invalid folder path.")

    # Option 2: Combine All
    if choice in ['2', '3']:
        # If choice is 3, we reuse the folder_path from option 1
        if choice == '2':
            folder_path = input("Enter root folder containing MD files to combine: ").strip()
        
        if os.path.exists(folder_path):
            # Automatically set output folder to '[FolderName] COMBINED' inside the root folder
            folder_name = os.path.basename(folder_path)
            output_folder = os.path.join(folder_path, f"{folder_name} COMBINED")
            print(f"Output will be saved to: {output_folder}")
            
            combine_all_md_files_recursive(folder_path, output_folder)
        else:
            print("Invalid folder path.")

if __name__ == "__main__":
    main()
