import subprocess
import sys
import os

def run_command(cmd, cwd=None):
    print(f"Running command: {' '.join(cmd)} (Cwd: {cwd})")
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    if rc != 0:
        print(f"Command failed with exit code: {rc}")
        sys.exit(rc)
    print("Command completed successfully!\n")

def main():
    # 1. Triplet Extraction
    print("=== Step 1: Running triplet extraction on HotpotQA test subset ===")
    run_command([sys.executable, 'hotpot_extraction_test.py'], cwd='code/preprocess')
    
    # 2. Run KG2RAG Distractor
    print("=== Step 2: Running KG2RAG Distractor pipeline ===")
    os.makedirs('output/hotpot', exist_ok=True)
    run_command([
        sys.executable, 'kg_rag_distractor.py',
        '--dataset', 'hotpotqa',
        '--data_path', '../data/hotpotqa/hotpot_dev_distractor_test.json',
        '--kg_dir', '../data/hotpotqa/kgs/extract_subkgs',
        '--result_path', '../output/hotpot/hotpot_dev_distractor_test_kgrag.json',
        '--model_name', 'llama3.2',
        '--top_k', '10'
    ], cwd='code')
    
    print("=== Pipeline Test Completed Successfully! ===")

if __name__ == '__main__':
    main()
