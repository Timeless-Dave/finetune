import os
import json
import csv
import subprocess
import shutil
import re
import difflib

# Try to import sentence-transformers, but have fallback if not available
USE_EMBEDDINGS = True
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    USE_EMBEDDINGS = False
    print("Warning: sentence-transformers not available. Using fallback text similarity instead.")

# Try to import PDF reading capability
PDF_SUPPORT = True
PDF_LIBRARY = None
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        PDF_SUPPORT = False
        print("Warning: No PDF library found. Install PyPDF2 or pdfplumber to read PDF files: pip install PyPDF2")

# ----------------------------
# CONFIGURATION
# ----------------------------
LOCAL_REPO_NAME = "temp_repo"
OUTPUT_JSONL = "fine_tune_data.jsonl"
OUTPUT_CSV = "fine_tune_data.csv"
APPEND_TO_EXISTING = False  # Set to True to append to existing output files

PROBLEM_KEYWORDS = ["problem", "question", "problems", "questions", "problemset"]
SOLUTION_KEYWORDS = ["solution", "answer", "solutions", "answers", "solutionset"]
SIMILARITY_THRESHOLD = 0.5  # Adjust if too many/few matches
# ----------------------------

def clone_repo(url, folder_name):
    # Try to remove existing folder, but with error handling for Windows permission issues
    if os.path.exists(folder_name):
        try:
            shutil.rmtree(folder_name)
        except PermissionError:
            print(f"Warning: Could not remove {folder_name} directory due to permissions.")
            print("Using existing repository folder...")
            return
        except Exception as e:
            print(f"Warning: {e}")
            print("Using existing repository folder...")
            return
    
    print(f"Cloning repo from {url} ...")
    subprocess.run(["git", "clone", url, folder_name])
    print("Repo cloned successfully.")

def find_folder_by_keywords(repo_path, keywords):
    # First try exact match for folder names
    for root, dirs, files in os.walk(repo_path):
        for d in dirs:
            if any(k.lower() in d.lower() for k in keywords):
                folder_path = os.path.join(root, d)
                # Check if the folder has files
                has_files = False
                for _, _, folder_files in os.walk(folder_path):
                    if folder_files:
                        has_files = True
                        break
                
                if has_files:
                    return folder_path
    
    # If no folder with files found, just return the first match
    for root, dirs, files in os.walk(repo_path):
        for d in dirs:
            if any(k.lower() in d.lower() for k in keywords):
                return os.path.join(root, d)
                
    return None

def extract_text_from_pdf(filepath):
    """Extract text from PDF file"""
    if not PDF_SUPPORT:
        return None
    
    text = ""
    try:
        if PDF_LIBRARY == "PyPDF2":
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        elif PDF_LIBRARY == "pdfplumber":
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting PDF {filepath}: {e}")
        return None
    
    return text.strip() if text.strip() else None

def is_readable_file(filepath):
    """Check if a file is readable (text file or PDF)"""
    # Define text file extensions
    text_extensions = {
        '.txt', '.md', '.py', '.java', '.cpp', '.c', '.js', '.html', '.css', 
        '.json', '.xml', '.yaml', '.yml', '.sql', '.sh', '.bat', '.ps1',
        '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
        '.r', '.m', '.pl', '.lua', '.dart', '.ts', '.jsx', '.tsx'
    }
    
    # Check file extension
    _, ext = os.path.splitext(filepath.lower())
    
    # Include PDFs if we have PDF support
    if ext == '.pdf' and PDF_SUPPORT:
        return True
        
    if ext in text_extensions:
        return True
    
    # For files without extension, try to read a small sample
    if not ext:
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(1024)
                # Check if it's mostly printable ASCII/UTF-8
                try:
                    sample.decode('utf-8')
                    # Count printable characters
                    printable_chars = sum(1 for c in sample if c < 128 and (c >= 32 or c in [9, 10, 13]))
                    return printable_chars / len(sample) > 0.7 if sample else False
                except UnicodeDecodeError:
                    return False
        except Exception:
            return False
    
    return False

def load_files(folder):
    data = {}
    if not folder or not os.path.exists(folder):
        return data
    
    skipped_files = 0
    pdf_files = 0
    text_files = 0
    
    # Walk through the directory and all subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            
            # Skip non-readable files
            if not is_readable_file(path):
                skipped_files += 1
                continue
            
            content = None
            _, ext = os.path.splitext(path.lower())
            
            try:
                if ext == '.pdf':
                    # Extract text from PDF
                    content = extract_text_from_pdf(path)
                    if content:
                        pdf_files += 1
                    else:
                        skipped_files += 1
                        continue
                else:
                    # Read as text file
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if content:
                            text_files += 1
                
                if content and len(content) > 10:  # Minimum content length
                    # Use relative path as key to maintain folder structure info
                    rel_path = os.path.relpath(path, folder)
                    data[rel_path] = content
                    
            except Exception as e:
                print(f"Error reading {path}: {e}")
                skipped_files += 1
    
    print(f"Found {len(data)} readable files in {folder}: {text_files} text files, {pdf_files} PDFs (skipped {skipped_files} files)")
    return data

def match_by_similarity(problems, solutions, threshold=0.5):
    """
    Matches each problem to the most similar solution using embeddings or fallback text similarity.
    """
    if not problems or not solutions:
        return [{"prompt": p, "completion": "SOLUTION NOT FOUND"} for p in problems.values()]

    problem_texts = list(problems.values())
    solution_texts = list(solutions.values())
    problem_files = list(problems.keys())
    solution_files = list(solutions.keys())
    
    labeled_data = []
    
    if USE_EMBEDDINGS:
        # Use sentence-transformers for better semantic matching
        model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & fast
        
        # Encode embeddings
        print("Encoding problem embeddings...")
        prob_embeddings = model.encode(problem_texts, convert_to_tensor=True)
        print("Encoding solution embeddings...")
        sol_embeddings = model.encode(solution_texts, convert_to_tensor=True)
        
        for idx, p_emb in enumerate(prob_embeddings):
            # compute cosine similarity
            sims = util.cos_sim(p_emb, sol_embeddings)[0]
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
            
            completion_text = solution_texts[best_idx] if best_score >= threshold else "SOLUTION NOT FOUND"
            labeled_data.append({
                "prompt": problem_texts[idx],
                "completion": completion_text
            })
    else:
        # Fallback: use simple text similarity with difflib
        print("Using fallback text similarity matching...")
        for idx, problem_text in enumerate(problem_texts):
            # Get filename and problem name without extension
            problem_file = problem_files[idx]
            problem_base = os.path.splitext(os.path.basename(problem_file))[0]
            best_match = None
            best_score = 0
            
            # Clean problem name for better matching (remove spaces, lowercase)
            clean_problem = ''.join(problem_base.lower().split())
            
            # First try to find matching filenames (without extension)
            for sol_idx, sol_file in enumerate(solution_files):
                sol_base = os.path.splitext(os.path.basename(sol_file))[0]
                clean_sol = ''.join(sol_base.lower().split())
                
                # Check if solution filename contains problem filename or vice versa
                # Also check without spaces and case insensitive
                if (problem_base in sol_base or sol_base in problem_base or
                    clean_problem in clean_sol or clean_sol in clean_problem):
                    best_match = solution_texts[sol_idx]
                    best_score = threshold + 0.1  # Ensure it passes threshold
                    print(f"Match found: {problem_base} -> {sol_base}")
                    break
            
            # If no filename match, use difflib for content similarity
            if not best_match:
                for sol_idx, sol_text in enumerate(solution_texts):
                    # Use sequence matcher for text similarity
                    similarity = difflib.SequenceMatcher(None, problem_text, sol_text).ratio()
                    if similarity > best_score:
                        best_score = similarity
                        best_match = sol_text
            
            completion_text = best_match if best_match and best_score >= threshold else "SOLUTION NOT FOUND"
            labeled_data.append({
                "prompt": problem_text,
                "completion": completion_text
            })

    return labeled_data

def save_jsonl(data, filename, append=False):
    # Check if we should append and file exists
    mode = "a" if append and os.path.exists(filename) else "w"
    
    repo_tag = os.getenv('CURRENT_REPO_NAME', 'unknown_repo')
    
    # Format data for OpenAI fine-tuning
    openai_data = []
    for item in data:
        # Skip items without valid solutions
        if item['completion'] == 'SOLUTION NOT FOUND':
            continue
            
        # Format according to OpenAI's requirements
        formatted_item = {
            "messages": [
                {
                    "role": "user", 
                    "content": item['prompt']
                },
                {
                    "role": "assistant", 
                    "content": item['completion']
                }
            ]
        }
        
        # Add metadata (OpenAI allows custom fields)
        formatted_item["custom_id"] = f"{repo_tag}_{item.get('edition', 'unknown')}_{len(openai_data)}"
        formatted_item["source_repo"] = repo_tag
        formatted_item["edition"] = item.get('edition', 'unknown')
        formatted_item["problem_file"] = item.get('problem_file', 'unknown')
        formatted_item["solution_file"] = item.get('solution_file', 'unknown')
        
        # Add solution variant if present
        if 'solution_variant' in item:
            formatted_item["solution_variant"] = item['solution_variant']
        
        openai_data.append(formatted_item)
    
    with open(filename, mode, encoding="utf-8") as f:
        for item in openai_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    action = "Appended to" if mode == "a" else "Saved"
    print(f"{action} OpenAI-compatible JSONL at {filename} ({len(openai_data)} valid entries)")

def truncate_text(text, max_length=200):
    """Truncate text for better CSV readability"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [TRUNCATED]"

def clean_text_for_csv(text):
    """Clean text to make it more readable in CSV"""
    # Replace multiple newlines with spaces
    text = ' '.join(text.split())
    # Remove special characters that might break CSV
    text = text.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
    return text

def save_csv(data, filename, append=False):
    # Check if we should append and file exists
    file_exists = os.path.exists(filename)
    mode = "a" if append and file_exists else "w"
    
    # Create readable CSV data
    csv_data = []
    repo_tag = os.getenv('CURRENT_REPO_NAME', 'unknown_repo')
    
    for item in data:
        # Clean and truncate content for readability
        prompt_preview = clean_text_for_csv(truncate_text(item['prompt']))
        completion_preview = clean_text_for_csv(truncate_text(item['completion']))
        
        csv_row = {
            'source_repo': repo_tag,
            'edition': item.get('edition', 'unknown'),
            'problem_file': item.get('problem_file', 'unknown'),
            'solution_file': item.get('solution_file', 'unknown'),
            'prompt_length': len(item['prompt']),
            'completion_length': len(item['completion']),
            'status': 'MATCHED' if item['completion'] != 'SOLUTION NOT FOUND' else 'NO_MATCH',
            'prompt_preview': prompt_preview,
            'completion_preview': completion_preview
        }
        
        # Add solution variant if present
        if 'solution_variant' in item:
            csv_row['solution_variant'] = item['solution_variant']
            
        csv_data.append(csv_row)
    
    # Define fieldnames - include solution_variant if any item has it
    fieldnames = [
        "source_repo", "edition", "problem_file", "solution_file", 
        "status", "prompt_length", "completion_length"
    ]
    
    # Add solution_variant if present in any item
    if any('solution_variant' in item for item in data):
        fieldnames.append("solution_variant")
        
    fieldnames.extend(["prompt_preview", "completion_preview"])
    
    with open(filename, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Only write header if creating a new file or not appending
        if not file_exists or not append:
            writer.writeheader()
            
        writer.writerows(csv_data)
    
    action = "Appended to" if mode == "a" else "Saved"
    print(f"{action} readable CSV at {filename}")

def check_repo_has_both(problems_path, solutions_path):
    """Check if the repo has both problem and solution folders"""
    if not problems_path and not solutions_path:
        print("Warning: No problem or solution folders detected.")
        return False
    if not problems_path:
        print("Warning: No problem folder found.")
        return False
    if not solutions_path:
        print("Warning: No solution folder found.")
        return False
    return True

def extract_ieee_repo_data(repo_path):
    """
    Extract data from IEEE repository structure where each edition has problemset/solutionset folders
    """
    all_data = []
    
    # Find all IEEE edition folders
    edition_folders = []
    for item in os.listdir(repo_path):
        item_path = os.path.join(repo_path, item)
        if os.path.isdir(item_path) and "ieee" in item.lower():
            edition_folders.append((item, item_path))
    
    print(f"Found {len(edition_folders)} IEEE edition folders")
    
    for edition_name, edition_path in edition_folders:
        print(f"\nProcessing {edition_name}...")
        
        # Look for problemset and solutionset folders in this edition
        problemset_path = None
        solutionset_path = None
        
        for folder in os.listdir(edition_path):
            folder_path = os.path.join(edition_path, folder)
            if os.path.isdir(folder_path):
                if any(keyword in folder.lower() for keyword in PROBLEM_KEYWORDS):
                    problemset_path = folder_path
                elif any(keyword in folder.lower() for keyword in SOLUTION_KEYWORDS):
                    solutionset_path = folder_path
        
        if not problemset_path or not solutionset_path:
            print(f"  Skipping {edition_name}: missing problemset or solutionset folder")
            continue
            
        print(f"  Found problemset: {os.path.basename(problemset_path)}")
        print(f"  Found solutionset: {os.path.basename(solutionset_path)}")
        
        # Load files from both folders
        solutions = load_files(problemset_path)  # Note: problemset contains solutions
        problems = load_files(solutionset_path)  # Note: solutionset contains problems
        
        print(f"  Loaded {len(problems)} problems and {len(solutions)} solutions")
        
        # Match by filename
        edition_data = match_by_filename(problems, solutions, edition_name)
        all_data.extend(edition_data)
        
        print(f"  Created {len(edition_data)} problem-solution pairs")
    
    return all_data

def match_by_filename(problems, solutions, edition_name):
    """
    Match problems to solutions based on exact filename matching
    """
    labeled_data = []
    
    for problem_file, problem_content in problems.items():
        problem_basename = os.path.splitext(os.path.basename(problem_file))[0]
        
        # Find all solutions with matching basename
        matching_solutions = []
        for solution_file, solution_content in solutions.items():
            solution_basename = os.path.splitext(os.path.basename(solution_file))[0]
            if problem_basename.lower() == solution_basename.lower():
                matching_solutions.append((solution_file, solution_content))
        
        if matching_solutions:
            # Create an entry for each matching solution
            for i, (solution_file, solution_content) in enumerate(matching_solutions):
                entry = {
                    "prompt": problem_content,
                    "completion": solution_content,
                    "edition": edition_name,
                    "problem_file": problem_file,
                    "solution_file": solution_file
                }
                
                # If multiple solutions, add suffix to distinguish
                if len(matching_solutions) > 1:
                    entry["solution_variant"] = i + 1
                
                labeled_data.append(entry)
                print(f"    Matched: {problem_basename} -> {os.path.basename(solution_file)}")
        else:
            print(f"    No solution found for: {problem_basename}")
    
    return labeled_data

def main():
    # Show capabilities
    print("=" * 60)
    print("GitHub Repository Data Extractor for Fine-Tuning")
    print("=" * 60)
    print(f"✅ Text file support: Enabled")
    print(f"{'✅' if PDF_SUPPORT else '❌'} PDF support: {'Enabled' if PDF_SUPPORT else 'Disabled (install PyPDF2: pip install PyPDF2)'}")
    print(f"{'✅' if USE_EMBEDDINGS else '❌'} Semantic matching: {'Enabled' if USE_EMBEDDINGS else 'Disabled (using fallback)'}")
    print("=" * 60)
    
    # Check if temp_repo already exists
    if os.path.exists(LOCAL_REPO_NAME):
        print(f"Found existing {LOCAL_REPO_NAME} directory")
        use_existing = input("Use existing repository? (y/n, default: y): ").strip().lower()
        if use_existing != 'n':
            print("Using existing repository...")
        else:
            # Get GitHub URL and clone
            github_url = input("Enter GitHub repository URL: ")
            repo_name = github_url.rstrip('/').split('/')[-1]
            os.environ['CURRENT_REPO_NAME'] = repo_name
            clone_repo(github_url, LOCAL_REPO_NAME)
    else:
        # Get GitHub URL and clone
        github_url = input("Enter GitHub repository URL: ")
        repo_name = github_url.rstrip('/').split('/')[-1]
        os.environ['CURRENT_REPO_NAME'] = repo_name
        clone_repo(github_url, LOCAL_REPO_NAME)
    
    # Ask if user wants to append to existing files
    append_choice = input("Append to existing files? (y/n, default: n): ").strip().lower()
    append_mode = append_choice == 'y'
    
    # Extract data using IEEE repository structure
    print("\nExtracting data from IEEE repository structure...")
    labeled_data = extract_ieee_repo_data(LOCAL_REPO_NAME)
    
    if not labeled_data:
        print("No problem-solution pairs found!")
        return
    
    print(f"\nTotal problem-solution pairs extracted: {len(labeled_data)}")
    
    # Save outputs with option to append
    save_jsonl(labeled_data, OUTPUT_JSONL, append=append_mode)
    save_csv(labeled_data, OUTPUT_CSV, append=append_mode)

    print("All done! You can now review the CSV for manual fixes if needed.")

if __name__ == "__main__":
    main()
